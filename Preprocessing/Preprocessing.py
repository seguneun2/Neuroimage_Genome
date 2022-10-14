"""
Preprocessing using nipype interface ants
#1. bias remove 
#2. resgistration < icbm 152 t1 template
#3. crop < optional

++
input_node 
 :  datapath list-> sub id , 

write_node 
 : save bias corrected file, registed file in working dir
   corrected_ {sub id}.nii. 
   registed_{sub id}.nii
"""


import os
import glob
import nipype.pipeline.engine as npe
import nipype.interfaces.utility as nutil
from nipype.interfaces import ants
import nipype.interfaces.io as nio  


### RAW_PATH
# ROOT_PATH = '/home/cps_lab/seungeun/ADNI/'
# nii_file_path_list = glob.glob(ROOT_PATH + '**/*.nii*', recursive=True)   
# t1w_files = nii_file_path_list[:3]

#### TEST




work_name = 'multi_test'
working_directory = os.getcwd()
ROOT_PATH = '/home/cps_lab/seungeun/MRI/preprocessing/test_raw_nii/'
t1w_files = glob.glob(ROOT_PATH + '**/*.nii')   
t1w_files = t1w_files[:2]
print(t1w_files)
ref_template = working_directory+'/Ref_template/mni_icbm152_t1_tal_nlin_sym_09c.nii'
ref_crop = working_directory + './Ref_template/ref_cropped_template.nii.gz'

#### multi processing
n_procs = 32


def crop_nifti(input_img, ref_crop):
    """
    :param input_img:
    :param crop_sagittal:
    :param crop_coronal:
    :param crop_axial:
    :return:
    """

    import nibabel as nib
    import os
    import numpy as np
    from nilearn.image import resample_img, resample_to_img
    from nibabel.spatialimages import SpatialImage

    basedir = os.getcwd()

    # resample the individual MRI onto the cropped template image
    crop_img = resample_to_img(input_img, ref_crop, force_resample=True)
    crop_img.to_filename(os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz'))

    output_img = os.path.join(basedir, os.path.basename(input_img).split('.nii')[0] + '_cropped.nii.gz')
    crop_template = ref_crop

    return output_img, crop_template

def get_input_fields():
        """"Specify the list of possible inputs of this pipelines.
        Returns:
        A list of (string) input fields name.
        """
        return ['t1w']


######### dataloader
# t1w_files = glob.glob('/home/cps_lab/seungeun/ADNI//**/*.nii')

def get_subject_id(file_path):
    sub_name= os.path.basename(file_path)


########## 
# Data Grabber


read_node = npe.Node(
            name="ReadingFiles",
            iterables=[
                ('t1w', t1w_files),   ## clinical file reader  : reaturn list of subject/session
                ],
            synchronize=True,
            interface=nutil.IdentityInterface(fields=get_input_fields()),
            n_procs=n_procs
            )

sub_id_node = npe.Node(
            interface=nutil.Function(
                input_names=['file_path'],
                output_names=['sub_id'],
                function=get_subject_id),
            name='subject_ID'
            )


# 1. Biascorrection

n4biascorrection_node = npe.Node(
            name='n4biascorrection',
            interface=ants.N4BiasFieldCorrection(
                dimension=3,
                save_bias=True,
                bspline_fitting_distance=600
                ),
            n_procs=n_procs
            )

# 2. `RegistrationSynQuick` by *ANTS*. It uses nipype interface.
ants_registration_node = npe.Node(
            name='antsRegistrationSynQuick',
            interface=ants.RegistrationSynQuick(),
            n_procs=n_procs
            )
# sub_name = os.path.basename(ants_registration_node.inputs.moving_image[0]).split('/ADNI/')[1].split('/')[0] 
ants_registration_node.inputs.fixed_image = ref_template
ants_registration_node.inputs.transform_type = 'a'
ants_registration_node.inputs.dimension = 3
ants_registration_node.inputs.num_threads = 8
# ants_registration_node.inputs.output_prefix = f'registed_{sub_name}'


# ants_registration_node.inputs.output_warped_image = 'transformWarped_{file_name}.nii.gz'



# 3. cropnifti
cropnifti = npe.Node(
            name='cropnifti',
            interface=nutil.Function(
                function=crop_nifti,
                input_names=['input_img', 'ref_crop'],
                output_names=['output_img', 'crop_template']
                )
            )
cropnifti.inputs.ref_crop = ref_crop


# write node ----> 폴더마다,,,,,,,되나....???????
write_node = npe.Node(
                name="Write",
                interface=nio.DataSink()
                )
write_node.inputs.base_directory = os.path.abspath(
                    f'{work_name}/output')
write_node.inputs.parameterization = False

###########################################
# workflow


"""
connect([(source, dest, [("source_output1", "dest_input1")]
"""

wf = npe.Workflow(name=work_name, base_dir=work_name)
wf.write_graph("workflow_graph.dot")
wf.connect([
        (read_node, n4biascorrection_node, [("t1w", "input_image")]),
        (n4biascorrection_node, ants_registration_node, [('output_image', 'moving_image')]),
        (ants_registration_node, write_node, [('warped_image', '@outfile_reg')]),
        (n4biascorrection_node, write_node, [('output_image', '@outfile_corr')]),
])

wf.run(plugin='MultiProc', plugin_args={'n_procs' : n_procs})

wf.write_graph("workflow_graph.dot")
from IPython.display import Image
Image(filename="./workflow_graph.png")


