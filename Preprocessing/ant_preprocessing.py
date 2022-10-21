"""
Preprocessing using nipype interface ants
#1. bias remove 
#2. resgistration < icbm 152 t1 template
#3. crop < optional

"""

import glob
import os
import concurrent.futures




def bias_corr(input_image):
    from nipype.interfaces.ants import N4BiasFieldCorrection

    """
    input_image : path of nii file --> raw_path  # ADNI/002_S_0295/MPR__GradWarp__B1_Correction__N3__Scaled_2/2006-04-18_08_20_30.0/I118671/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.nii
    
    create {sub_id}_correted.nii, {sub_id}_bias.nii :  using ant.N4BiasFieldCorrection

    return : created nii file path object , sub_id
    """
    sub_id = input_image.split("ADNI/")[1].split("/")[0]
    # print(sub_id)
    bias_corr = N4BiasFieldCorrection(
                dimension=3,
                save_bias=True,
                bspline_fitting_distance=600
                )

    bias_corr.inputs.input_image = input_image
    bias_corr.inputs.output_image = os.getcwd()+f"/Bias_corrected/{sub_id}_corrected.nii"
    bias_corr.inputs.num_threads = num_thread
    print(bias_corr.cmdline)
    bias_corr.run()
    
    return bias_corr._list_outputs()['output_image']


def Regsitration(input_image):
    """
    input_image : path of bias corrected file path 
    using ants.Registration , ref template 
    
    Save wraped_image to dst dir
    
    """
    from nipype.interfaces.ants import RegistrationSynQuick
    import shutil

    sub_id = os.path.basename(input_image).split("_correct")[0]

    ref_template = os.getcwd()+'/Ref_template/mni_icbm152_t1_tal_nlin_sym_09c.nii'
    dst_dir = os.path.join(os.getcwd() , 'Registration_BiasCorr/')
    inversed_dst_dir = os.path.join(os.getcwd() , 'Registration_BiasCorr_inversed/')

    for path_dir in [dst_dir, inversed_dst_dir]:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)

        
    reg = RegistrationSynQuick()
    reg.terminal_output = "none"

    reg.inputs.fixed_image = ref_template
    reg.inputs.moving_image = input_image
    reg.inputs.num_threads = num_thread
    reg.inputs.output_prefix = sub_id

    
    ###### prepifx, termenial output  ,, warped_image만 사용,
    
    reg.run()
    print(f"Registration {sub_id}")
    """
    {'warped_image': '/home/cps_lab/seungeun/MRI/preprocessing/transformWarped.nii.gz',
    'inverse_warped_image': '/home/cps_lab/seungeun/MRI/preprocessing/transformInverseWarped.nii.gz',
    'out_matrix': '/home/cps_lab/seungeun/MRI/preprocessing/transform0GenericAffine.mat',
    'forward_warp_field': '/home/cps_lab/seungeun/MRI/preprocessing/transform1Warp.nii.gz',
    'inverse_warp_field': '/home/cps_lab/seungeun/MRI/preprocessing/transform1InverseWarp.nii.gz'}
    
    """
    shutil.move(reg._list_outputs()['warped_image'], dst_dir + f"{sub_id}.nii.gz")
    shutil.move(reg._list_outputs()['inverse_warped_image'], inversed_dst_dir + f"Inverse_{sub_id}.nii.gz")

    shutil.move(reg._list_outputs()['out_matrix'], os.getcwd()+'out_matrix.mat' )
    shutil.move(reg._list_outputs()['forward_warp_field'], os.getcwd()+'/forward_warp_filed.nii.gz')
    shutil.move(reg._list_outputs()['inverse_warp_field'],os.getcwd()+'/inverse_warp_filer.nii.gz')


if __name__== "__main__":
    # file_dir = '/home/cps_lab/seungeun/ADNI/'
    import multiprocessing
    
    print("os.getcwd() : ", os.getcwd())
    file_dir = '/home/cps_lab/seungeun/MRI/Preprocessing/Bias_corrected/'
    file_list = [x for x in glob.glob(file_dir + '*.nii*')]
    num_thread = 128
    #print(file_list)
    
    pool = multiprocessing.Pool(processes=48)
    pool.map(Regsitration, file_list)
    pool.close()
    pool.join()
