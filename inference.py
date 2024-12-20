"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-test-phase | gzip -c > example-algorithm-test-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!

"""
from pathlib import Path
import json
from glob import glob
import shutil
import SimpleITK as sitk
import os
import tempfile
from itertools import chain

PATH_DEEPISLES = os.path.join(os.getcwd(), 'DeepIsles')
import sys
sys.path.append(PATH_DEEPISLES)
from DeepIsles.src.isles22_ensemble import IslesEnsemble


#INPUT_PATH = Path("/input")
#OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")

class predict():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = False  # False for running the docker!
        if self.debug:
            self._input_path = Path('/home/edelarosa/Documents/git/deepisles_gc/test/input')
            self._output_path = Path('/home/edelarosa/Documents/datasets/example_dwi/test_me_gc')
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._algorithm_output_thumbnail_path = self._output_path /'stroke-lesion-segmentation-thumbnail.png'

            self._output_file = self._output_path / 'results.json'
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._algorithm_output_thumbnail_path = self._output_path /'stroke-lesion-segmentation-thumbnail'

            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []

    def predict(self, input_data_paths):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image', 'dwi_json', 'adc_json', 'flair_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image_path, adc_image_path, flair_image_path, deepisles_config_path = input_data_paths['dwi_image_path'],\
                                            input_data_paths['adc_image_path'],\
                                            input_data_paths['flair_image_path'], \
                                            input_data_paths['deepisles_config_path']

        with open(deepisles_config_path, 'r') as file:
            run_config = json.load(file)  # Parse JSON into a Python dictionary
        skull_strip = bool(run_config["skull_strip"])
        # predict with deepisles
        stroke_segm = IslesEnsemble()
        deepisles_out_path = tempfile.mkdtemp(prefix="tmp", dir="/tmp")

        if self.debug:
            weights_dir = os.path.join(os.getcwd(), 'test', 'opt', 'ml', 'model', 'weights')
        else:
            weights_dir = os.path.join(str(MODEL_PATH), 'weights')
        stroke_segm.predict_ensemble(ensemble_path=PATH_DEEPISLES,
                                     input_dwi_path=str(dwi_image_path),
                                     input_adc_path=str(adc_image_path),
                                     input_flair_path=str(flair_image_path),
                                     output_path=deepisles_out_path,
                                     fast=False,
                                     save_team_outputs=False,
                                     skull_strip=skull_strip,
                                     results_mni=False,
                                     weights_dir=weights_dir)

        output_msk_path = os.path.join(deepisles_out_path, 'lesion_msk.nii.gz')
        output_png_file = os.path.join(deepisles_out_path, 'output_screenshot.png')

        #################################### End of your prediction method. ############################################
        ################################################################################################################

        return output_msk_path, output_png_file

    def process_isles_case(self, input_data_paths, input_filename):

        # Segment images.
        output_msk_path, output_png_file = self.predict(input_data_paths) # TODO check if .nii outputs are ok

        # origin, spacing, direction = input_data_paths['dwi_image'].GetOrigin(),\
        #                              input_data_paths['dwi_image'].GetSpacing(),\
        #                              input_data_paths['dwi_image'].GetDirection()

        # Build the itk object.
        # output_image = SimpleITK.GetImageFromArray(prediction)
        # output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
            os.makedirs(str(self._algorithm_output_thumbnail_path))

        output_image_path = (self._algorithm_output_path / input_filename).with_name(
            f"{Path(input_filename).stem}-msk.mha")

        #output_thumbnail_path = self._algorithm_output_thumbnail_path / 'stroke-lesion-segmentation-thumbnail.png'

        # export output as .mha
        image = sitk.ReadImage(output_msk_path)
        sitk.WriteImage(image, str(output_image_path))

        #shutil.copyfile(output_msk_path, output_image_path) #copy tmp file to GC required location
        shutil.copyfile(output_png_file, self._algorithm_output_thumbnail_path) #copy tmp file to GC required location

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {"outputs": [dict(type="Image", slug="stroke-lesion-segmentation",
                                                 filename=str(output_image_path.name)),
                                       dict(type="Thumbnail", slug="stroke-lesion-segmentation-thumbnail",
                                            filename=str(output_thumbnail_path.name))],

                                       "inputs": [dict(type="Image", slug="dwi-brain-mri",
                                           filename=input_filename)]}

            self._case_results.append(json_result)
            self.save()

    def load_isles_case(self):
        """ Loads the 3 inputs """

        # Get MR data paths.
        dwi_image_path = self.get_file_path(slug='dwi-brain-mri', filetype='image')
        adc_image_path = self.get_file_path(slug='adc-brain-mri', filetype='image')
        flair_image_path = self.get_file_path(slug='flair-brain-mri', filetype='image')
        deepisles_config_path = self.get_file_path(slug='deepisles_settings', filetype='json')
        #input_data = {'dwi_image': SimpleITK.ReadImage(str(dwi_image_path)), 'dwi_json': json.load(open(dwi_json_path)),
                      # 'adc_image': SimpleITK.ReadImage(str(adc_image_path)), 'adc_json': json.load(open(adc_json_path)),
                      # 'flair_image': SimpleITK.ReadImage(str(flair_image_path)), 'flair_json': json.load(open(flair_json_path))}

        input_data_paths = {'dwi_image_path': dwi_image_path, 'adc_image_path': adc_image_path,
                            'flair_image_path': flair_image_path, "deepisles_config_path":deepisles_config_path}

        # Set input information.
        input_filename = str(dwi_image_path).split('/')[-1].split('.')[0]
        return input_data_paths, input_filename

    def get_file_path(self, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':

            file_list = list(chain(
        (self._input_path / "images" / slug).glob("*.mha")#,
#                 (self._input_path / "images" / slug).glob("*.nii"),
 #                (self._input_path / "images" / slug).glob("*.nii.gz"),
                )
            )

        elif filetype == 'json':
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        input_data_paths, input_filename = self.load_isles_case()
        self.process_isles_case(input_data_paths, input_filename)

if __name__ == "__main__":
    predict().process()
