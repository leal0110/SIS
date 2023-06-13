import os.path

from torch.utils.data import DataLoader
from evaluation.eval import eval_one_result
import dataloaders.pascal as pascal
# import dataloaders.ct as ct
# import dataloaders.sstem as sstem
# import dataloaders.mri as mri

exp_root_dir = './'

method_names = []
method_names.append('t_f')

if __name__ == '__main__':

    # Dataloader
    dataset = pascal.VOCSegmentation(transform=None, retname=True)
    # dataset = ct.Segmentation(transform=None, retname=True)
    # dataset = sstem.Segmentation(transform=None, retname=True)
    # dataset = mri.Segmentation(transform=None, retname=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Iterate through all the different methods
    for method in method_names:
        for ii in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            results_folder = os.path.join(exp_root_dir, method, 'Results_new')
            filename = os.path.join(exp_root_dir, method, 'eval_results', method.replace('/', '-') + '.txt')
            if not os.path.exists(os.path.join(exp_root_dir, method, 'eval_results')):
                os.makedirs(os.path.join(exp_root_dir, method, 'eval_results'))
            jaccards = eval_one_result(dataloader, results_folder, mask_thres=ii)
            val = jaccards["all_jaccards"].mean()


            print(f'mask_thres : {ii}')
            print('Result for {} : {}'.format(method, str.format("{0:.4f}", 100 * val)))
            with open(filename, 'a') as f:
                f.write(str(ii) + ' : ' + str(val) + '\n')