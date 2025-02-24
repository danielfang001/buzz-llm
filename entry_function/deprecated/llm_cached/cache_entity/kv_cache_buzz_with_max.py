from entry_function.deprecated.llm_cached.cache_data_access.kv_crud_manager import *

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

DIM_TO_SLICE_WITH_LOCAL_MAX = {
    1: slice1d_with_local_max,
    2: slice2d_with_local_max,
    3: slice3d_with_local_max,
}


class FixedStrideKVCacheWithMax:
    def __init__(
            self,
            sink_size=4,
            window_size=180,
            stride_size=4,
            k_seq_dim=2,
            v_seq_dim=2,
            sample_threshold=180
    ):
        print(
            f"FixedStrideKVCache with local max, sink size, stride size, sample threshold, window size: "
            f"{sink_size}, "
            f"{stride_size}, "
            f"{sample_threshold}, "
            f"{window_size}")
        self.sink_size = sink_size
        self.window_size = window_size
        self.stride_size = stride_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.k_slice_with_max = DIM_TO_SLICE_WITH_LOCAL_MAX[k_seq_dim]
        self.v_slice_with_max = DIM_TO_SLICE_WITH_LOCAL_MAX[v_seq_dim]
        self.sample_threshold = sample_threshold
        self.cur_sample_size = 0

    def get_sink_other_window(self, new_coming_words):
        new_coming_size = new_coming_words[0][0].size(self.k_seq_dim)
        if new_coming_size <= self.sink_size:
            return new_coming_words, [], []
        elif new_coming_size <= self.sink_size + self.window_size:
            return (
                [
                    [
                        self.k_slice(k, 0, self.sink_size),
                        self.v_slice(v, 0, self.sink_size),
                    ]
                    for k, v in new_coming_words
                ],
                [],
                [
                    [
                        self.k_slice(k, self.sink_size),
                        self.v_slice(v, self.sink_size),
                    ]
                    for k, v in new_coming_words
                ],
            )
        else:
            return (
                [
                    [
                        self.k_slice(k, 0, self.sink_size),
                        self.v_slice(v, 0, self.sink_size),
                    ]
                    for k, v in new_coming_words
                ],
                [
                    [
                        self.k_slice(k, self.sink_size, -self.window_size),
                        self.v_slice(v, self.sink_size, -self.window_size),
                    ]
                    for k, v in new_coming_words
                ],
                [
                    [
                        self.k_slice(k, -self.window_size),
                        self.v_slice(v, -self.window_size),
                    ]
                    for k, v in new_coming_words
                ],
            )

    def concat_three_parts(self, one, two, three):
        layer_num = max(len(one), len(two), len(three))
        concatenation = []
        for i in range(layer_num):
            part_one_0 = one[i][0] if i < len(one) else None
            part_two_0 = two[i][0] if i < len(two) else None
            part_three_0 = three[i][0] if i < len(three) else None

            part_one_1 = one[i][1] if i < len(one) else None
            part_two_1 = two[i][1] if i < len(two) else None
            part_three_1 = three[i][1] if i < len(three) else None

            # 拼接part 0
            parts_0 = [p for p in [part_one_0, part_two_0, part_three_0] if p is not None]
            concatenated_0 = torch.cat(parts_0, dim=self.k_seq_dim) if parts_0 else None

            # 拼接part 1
            parts_1 = [p for p in [part_one_1, part_two_1, part_three_1] if p is not None]
            concatenated_1 = torch.cat(parts_1, dim=self.k_seq_dim) if parts_1 else None

            concatenation.append([concatenated_0, concatenated_1])

        return concatenation

    def sample_extract(self, evicted, stride):
        return \
            [
                [
                    self.k_slice(k, start=0, step=stride),
                    self.v_slice(v, start=0, step=stride),
                ]
                for k, v in evicted
            ]

    @staticmethod
    def de_repeat_q(hidden_states, n_de_rep):
        batch, num_of_q_heads, s_len, head_dim = hidden_states.shape
        if num_of_q_heads % n_de_rep != 0:
            raise ValueError(f"Cannot chunk because {num_of_q_heads} % {n_de_rep}")
        chunks = torch.chunk(hidden_states, chunks=n_de_rep, dim=1)
        return torch.sum(torch.stack(chunks, dim=0), dim=0)

    def deal_new_sample_with_local_max(self, k, v, attn_score, probability=0.5):
        # attn有32个头，去除为4个头
        attn_score = self.de_repeat_q(attn_score, 8)
        # hh_score是一个头 * token len的矩阵，代表了在每个头下，每个token收获的注意力
        hh_score = attn_score.sum(0).sum(1)
        # 因为local max抽样的部分只有new evicted，要去掉sink, old sample, window
        hh_score = hh_score[:, self.sink_size + self.cur_sample_size:-self.window_size]
        return [self.k_slice_with_max(x=k, start=0, step=self.stride_size, attn_scores=hh_score, probability=1),
                self.v_slice_with_max(x=v, start=0, step=self.stride_size, attn_scores=hh_score, probability=1)]

    def __call__(self, current_words, attn_scores):
        if current_words is None:
            raise ValueError("new_coming_words cannot be None")
        else:
            cur_cache_size = current_words[0][0].size(self.k_seq_dim)
            if cur_cache_size <= self.window_size + self.sink_size:
                return current_words
            sink, other, window = self.get_sink_other_window(current_words)
            other_size = other[0][0].size(self.k_seq_dim)
            if other_size <= self.sample_threshold:
                return current_words
            old_sample = [
                [
                    self.k_slice(k, 0, self.cur_sample_size),
                    self.v_slice(v, 0, self.cur_sample_size)
                ]
                for k, v in other
            ]
            smaller_stride = (self.stride_size + 1) // 2
            old_sample = self.sample_extract(old_sample, smaller_stride)
            evicted = [
                [
                    self.k_slice(k, self.cur_sample_size),
                    self.v_slice(v, self.cur_sample_size)
                ]
                for k, v in other
            ]

            # TODO 都有22层，要分层处理， 但有一个问题，k和v只处理新来的
            new_sample = []
            for i in range(0, len(evicted)):
                # 1 * 4 * (X - sink - window - old) * 64
                cur_k = evicted[i][0]
                # 1 * 4 * (X - sink - window - old) * 64
                cur_v = evicted[i][1]
                # 1 * 32 * X * X
                cur_attn_score = attn_scores[i].detach().clone()
                new_sample.append(self.deal_new_sample_with_local_max(cur_k, cur_v, cur_attn_score))
            cur_sample = self.concat_three_parts(old_sample, new_sample, [])
            self.cur_sample_size = cur_sample[0][0].size(self.k_seq_dim)
            return self.concat_three_parts(sink, cur_sample, window)

