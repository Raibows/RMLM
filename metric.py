
class Metric():
    def __init__(self, save_file=None):
        self.records = {}
        self.save_file = save_file
        self.best_path = {}

    def add_record(self, metric_name, epoch, value):
        if epoch not in self.records:
            self.records[epoch] = {}
        self.records[epoch][metric_name] = value

    def save(self):
        if self.save_file:
            with open(self.save_file, 'w') as file:
                for ep, value in self.records.items():
                    temp = f"epoch {ep:3d}"
                    for k, v in value.items():
                        temp += f"  {k} {v:.5f}"
                    file.write(temp)
                    file.write('\n')

    def __repr__(self):
        return str(self.records)

    def find_best(self, metric_name='loss', lower_is_better=True, top=1):
        temp = sorted(self.records.items(), key=lambda item: item[1][metric_name], reverse=not lower_is_better)
        return dict(temp[:top])