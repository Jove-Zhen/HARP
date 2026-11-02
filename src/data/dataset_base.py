import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class RecDataset:
    """Base recommendation dataset with leave-one-out split (TSV .inter format)"""

    def __init__(self, data_path: str, seed: int = 42):
        import pandas as pd
        self.data_path = Path(data_path)
        self.name = self.data_path.stem
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.df = pd.read_csv(data_path, sep='\t')
        self.n_users = self.df['user_id'].max() + 1
        self.n_items = self.df['item_id'].max() + 1
        self.n_interactions = len(self.df)

        self.user_items = defaultdict(list)
        for _, row in self.df.iterrows():
            self.user_items[int(row['user_id'])].append(int(row['item_id']))

        self._sort_by_timestamp()
        self.train_data, self.valid_data, self.test_data = self._leave_one_out_split()

        self.instructions = None
        self.reviews = None
        self.item_metadata = None
        self.ranked_lists = None

    def _sort_by_timestamp(self):
        import pandas as pd
        sorted_user_items = {}
        for user_id, items in self.user_items.items():
            user_df = self.df[self.df['user_id'] == user_id].sort_values('timestamp')
            sorted_user_items[user_id] = user_df['item_id'].tolist()
        self.user_items = sorted_user_items

    def _leave_one_out_split(self) -> Tuple[Dict, Dict, Dict]:
        train_data, valid_data, test_data = {}, {}, {}
        for user_id, items in self.user_items.items():
            if len(items) < 3:
                continue
            train_data[user_id] = items[:-2]
            valid_data[user_id] = items[-2]
            test_data[user_id] = items[-1]
        return train_data, valid_data, test_data

    def get_user_train_items(self, user_id: int) -> List[int]:
        return self.train_data.get(user_id, [])

    def get_user_history(self, user_id: int, split: str = 'test') -> List[int]:
        items = self.train_data.get(user_id, []).copy()
        if split in ['valid', 'test'] and user_id in self.valid_data:
            items.append(self.valid_data[user_id])
        if split == 'test' and user_id in self.test_data:
            items.append(self.test_data[user_id])
        return items


    def load_instructions(self):
        import pandas as pd
        instruction_path = self.data_path.parent / f"{self.data_path.stem}.instruction"
        if not instruction_path.exists():
            print(f"Warning: {instruction_path} not found")
            return
        df = pd.read_csv(instruction_path, sep='\t')
        self.instructions = {}
        for _, row in df.iterrows():
            self.instructions[int(row['user_id'])] = {
                'instruction': str(row['instruction']),
                'persona': str(row['persona']) if 'persona' in row else ''
            }
        print(f"Loaded instructions for {len(self.instructions)} users")

    def load_item_metadata(self):
        import pandas as pd
        meta_path = self.data_path.parent / f"{self.data_path.stem}.meta"
        if not meta_path.exists():
            print(f"Warning: {meta_path} not found")
            return
        self.item_metadata = {}
        for chunk in pd.read_csv(meta_path, sep='\t', chunksize=10000):
            for _, row in chunk.iterrows():
                item_id = int(row['item_id'])
                self.item_metadata[item_id] = {
                    'asin': str(row['asin']) if 'asin' in row else '',
                    'title': str(row['title']) if 'title' in row else '',
                    'description': str(row['description']) if 'description' in row else ''
                }
        print(f"Loaded metadata for {len(self.item_metadata)} items")
