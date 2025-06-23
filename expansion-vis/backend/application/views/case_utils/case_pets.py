from global_parameter import work_dir

class Case:
    def __init__(self, name, step):
        # Initialize the case with a name and step, then load data paths
        self.step = step
        self.name = name
        self.get_data()

    def run(self, use_buffer=False):
        print("self.step", self.step)
        if use_buffer:
            self.get_data()
            # Update data paths in the specified base directory
            print("Data has been updated in base_dir", work_dir + '/demo/' +
                  self.name + '/step' + str(self.step))
        else:
            # No changes made
            pass

    def get_data(self):
        base_dir = work_dir + '/demo/' + self.name + '/step' + str(self.step)
        print(base_dir)
        # Update file paths based on the current step
        self.metric_dir = base_dir + '/metric.json'
        self.original_data_dir = base_dir + '/pets/train'
        self.tsne_embedding_dir = base_dir + '/embeddings.json'
        self.image_caption_json = base_dir + '/images_info.json'
        self.tree_cut_json = base_dir + '/treecut.json'

        self.data_dir = base_dir + '/origin0510_scale20_strength05'

        self.word_frequency_json = base_dir + '/word_frequency.json'
        self.prompt_json = base_dir + '/prompts.json'
        self.words = base_dir + '/words.json'
        self.images = base_dir + '/images.json'

    def update(self, step):
        # Update the current step and reload data paths with buffering
        self.step = step
        self.run(use_buffer=True)

    def change_case(self, name):
        # Change the case name and reload data paths with buffering
        self.name = name
        self.run(use_buffer=True)


PetsCase = Case('pets', '3')
