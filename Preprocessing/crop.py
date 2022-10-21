ref_crop = '/home/cps_lab/seungeun/MRI/Preprocessing/Ref_template/ref_cropped_template.nii.gz'
import os
def crop_nifti(input_img, ref_crop = ref_crop):
    """
    :param input_img:
    :param crop_sagittal:
    :param crop_coronal:
    :param crop_axial:
    :return:
    """

    import nibabel as nib
    import numpy as np
    from nilearn.image import resample_img, resample_to_img
    from nibabel.spatialimages import SpatialImage

    basedir = os.getcwd() + '/cropped'
    if os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)


    crop_img = resample_to_img(input_img, ref_crop, force_resample=True)
    crop_img.to_filename(os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz'))

    output_img = os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz')



if __name__ == "__main__":
    import multiprocessing

    print("os.getcwd() : ", os.getcwd())
    file_dir = '/home/cps_lab/seungeun/MRI/Preprocessing/Registration_BiasCorr'
    file_list = [os.path.join(file_dir, _) for _ in os.listdir(file_dir) if _.endswith(".nii.gz")]

    print(file_list)
    num_thread = 128
    
    pool = multiprocessing.Pool(processes=48)
    pool.map(crop_nifti, file_list)
    pool.close()
    pool.join()
