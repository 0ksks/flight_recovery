from collections import defaultdict
import torch as th


def get_test(seed=0):
    th.manual_seed(seed)
    row = th.randint(5, 10, (1,))
    col = th.randint(row, row + 5, (1,))
    return th.randint(1, col + 1, (row, col)).to(th.float)


class MultiAct:
    @staticmethod
    def __get_row_rank(input_tensor: th.Tensor) -> th.Tensor:
        return th.sort(input_tensor, dim=1).indices

    @staticmethod
    def __get_col_rank(input_tensor: th.Tensor) -> th.Tensor:
        return th.sort(input_tensor, dim=0).indices

    @staticmethod
    def __check_conflict(
        row_rank: th.Tensor,
        col_rank: th.Tensor,
        max_index: th.Tensor,
    ) -> dict:
        value_indices = row_rank[range(row_rank.size(0)), max_index].tolist()
        conflict_dict = defaultdict(list)
        for idx, val in enumerate(value_indices):
            conflict_dict[val].append(idx)

        conflict_dict = {k: v for k, v in conflict_dict.items() if len(v) > 1}

        for k, v in conflict_dict.items():
            val_col_rank = col_rank[range(col_rank.size(0)), max_index]
            save_mask = th.zeros(max_index.size(0)).to(th.int)
            save_mask[v] = 1
            val_col_rank *= save_mask
            reserve_val = val_col_rank.argmax()
            v.remove(reserve_val)
            conflict_dict[k] = v

        return conflict_dict

    @staticmethod
    def __transport_lower(
        row_rank: th.Tensor,
        col_rank: th.Tensor,
        max_index: th.Tensor,
    ):
        conflict_dict = MultiAct.__check_conflict(row_rank, col_rank, max_index)
        if conflict_dict:
            for conflict_pos_list in conflict_dict.values():
                for conflict_pos in conflict_pos_list:
                    curr_idx = max_index[conflict_pos]
                    selected_idx = row_rank[range(row_rank.size(0)), max_index].tolist()
                    while row_rank[conflict_pos][curr_idx].item() in selected_idx:
                        curr_idx -= 1
                    max_index[conflict_pos] = curr_idx
        return max_index

    @staticmethod
    def test(seed):
        test = get_test(seed)
        row_rank = MultiAct.__get_row_rank(test)
        col_rank = MultiAct.__get_col_rank(test)
        max_index = th.tensor(
            [
                test.size(1) - 1,
            ]
            * test.size(0)
        )
        conflict_dict = MultiAct.__check_conflict(row_rank, col_rank, max_index)
        max_index = MultiAct.__transport_lower(row_rank, col_rank, max_index)

    @staticmethod
    def decide(action_score: th.Tensor) -> th.Tensor:
        row_rank = MultiAct.__get_row_rank(action_score)
        col_rank = MultiAct.__get_col_rank(action_score)
        max_index = th.tensor(
            [
                action_score.size(1) - 1,
            ]
            * action_score.size(0)
        )
        max_index = MultiAct.__transport_lower(row_rank, col_rank, max_index)
        return max_index
