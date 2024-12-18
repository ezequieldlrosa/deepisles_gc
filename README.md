This is an example algorithm that can be uploaded to Grand Challenge. 

The algorithm takes a `generic-medical-image` as input and produces a dummy `results-json` file as output. 
For demonstration purposes, it loads dummy model weights from `opt/ml/model`. The corresponding model is uploaded to the algorithm on Grand Challenge via the `Models` tab. Grand Challenge expects a `tar.gz` file and extracts the contents of that tar archive to `opt/ml/model` at runtime. What the tar archive contains exactly is up to the algorithm developer. For testing purposes, wwe use an empty `.pth` file.
When preparing your model weights for upload to Grand Challenge, we recommend you create the tarball with: `tar -czvf algorithmmodel.tar.gz -C /path/to/algorithmmodel .`

To locally test the algorithm, run `./test_run.sh`.

To build and save the image for upload to Grand Challenge, run `./save.sh`
