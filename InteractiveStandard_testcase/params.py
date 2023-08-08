from pathlib import Path


# Data params
dataloader_num_workers = 1

image_dir = Path('test-task/clusters/')
csv_dir = Path('test-task/clusters.csv')
model_folder = "saved_models"


data_cleaning = {
    'f75f7082d62a44e7bfd373532877c9a6': 36,
    '552656baa25848009b9eff3dfbedd15c': 10,
    '96ebf9d10efa4804a5580f4d55e64d38': 16,
    '3b9ba5ab3fce43c2bca84ee29e77c203': 39,
    '726df8d4f65f4855849925746d01f0ae': 27,
    '325cdb0bf6694d879a0f3518178feb60': 29,
    'a1a79389069f4ddd86bfbdec7078ed02': 19,
    '8d1c755d31a6484686c4a98ad45ab0e7': 29,
    '499c8686053d47ca8395c4f5b7780bd0': 37,
    '160dfcfe3c4e4623a3e9c360da87891c': 19,
    '6b9c4d2192c74d82bbe02d52c5b1e007': 16,
    '9fdbdc45cb7b4e53b0efaab769f7e224': 18,
    '403446096ccf434f9e096b8291c40ac9': 18,
    '23db675a000249498062844cdb31b76f': 18,
    'd6367566a1dc445b87600b8c98ea5402': 18,
    '980f8122f0db49cc94eef626826f3614': 22,
    'd9d0792cefa74c55bdef834be6d34653': 20,
    '670eb7df1cf54be09e18bc13496a50f1': 36,
    '427ee587bd854410ada20b619b186f2b': 26,
    '7bde72d406024ec0b7e72ca9e1edb311': 27,
    '334b17efee804bfd82bb4189c3331ddb': 20,
    'd94c01292d6f4fc19873234ccd85f091': 21,
    '68b0cc47f2034ec98a3db6bf8b9df8e5': 13,
    'afd68eec35824484b63a730ee8a19bef': 16,
    '9e3d20f8c75f4dbea6085c514c4afb48': 7
}


# Leaning params
num_epochs = 10
model_name ='EfficientNet_B3'
image_size = (320, 300) # specific size for EfficientNet_B3
embedding_size = 256 
batch_size = 12

lr = 0.00001
weight_decay = 0.0001

arcface_margin = 0.3
arcface_scale = 5
samples_per_class = 5

# Inference params
trunk_path = 'saved_models/trunk_best19.pth'
embedder_path = 'saved_models/embedder_best19.pth'


