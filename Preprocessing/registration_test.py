import multiprocessing
import os
import glob
from re import M
import subprocess
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import concurrent.futures


# scr_path_list = [x for x in glob.glob('/home/cps_lab/seungeun/ADNI/**/*.nii',recursive=True)]
ref_path = '/home/cps_lab/seungeun/MRI/preprocessing/Ref_template/mni_icbm152_t1_tal_nlin_sym_09c.nii'


def registration(src_path, ref_path = ref_path):
    #sub_id = src_path.split('ADNI/')[1].split('/')[0]
    sub_id = os.path.basename(src_path).split(".nii")[0]
    dst_dir = '/home/cps_lab/seungeun/MRI/preprocessing/Resampling_registration/'
    dst_path = dst_dir + sub_id

    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-bins", "256", "-cost", "corratio", "-searchrx", "0", "0",
               "-searchry", "0", "0", "-searchrz", "0", "0", "-dof", "12",
               "-interp", "spline"] 
    """option : """
    subprocess.call(command, stdout=open(os.devnull, "r"),
                    stderr=subprocess.STDOUT)
    print(f"{sub_id} registration finished. ")
    return


# print(scr_path_list)'
if __name__ == "__main__":
    scr_path_list = [x for x in glob.glob('/home/cps_lab/seungeun/MRI/preprocessing/Resampling/*.nii.gz',recursive=True)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        executor.map(registration, scr_path_list)
