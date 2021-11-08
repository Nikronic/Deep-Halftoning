# Deep Halftoning
"Deep Context-Aware Descreening and Rescreening of Halftone Images" paper implementation.

# WIP
Numbers in paper are not reproducible at the moment.

# Details
This project pertains automated Descreening process. Descreening is the task that we try to reconstruct the [halftoned](https://en.wikipedia.org/wiki/Halftone) image (which is the mandatory process to interact images with printers, scanners, monitors, etc) meanwhile reducing the amount of data loss.

* First and the only fully open source implementation of this paper, in PyTorch
* The implementation can be divided into below separate projects:
  * [CoarseNet](https://github.com/Nikronic/CoarseNet): Modified version of U-Net architecture to work as a low-pass filter to remove halftone patterns
  * [DetailsNet](https://github.com/Nikronic/DetailsNet): A deep CNN generator and two discriminators which are trained simultaneously to improve image quality
  * [EdgeNet](https://github.com/Nikronic/EdgeNet): A simple CNN model to extract Canny edge features to preserve details
  * [ObjectNet](https://github.com/Nikronic/ObjectNet): Modified version of "Pyramid Scene Parsing Network" to only return 25 major classified segments out of 150
  * [Halftoning-Algorithms](https://github.com/Nikronic/Halftoning-Algorithms): Implementation of some of the halftone algorithms provided in most recent digital color haltoning books as ground truth
  * [Places365-Preprocessing](https://github.com/Nikronic/Places365-Preprocessing): A custom and extendable implementation of Dataset to handle lazy loading of a huge data functionality

# Reference
## Paper Authors
Tae-hoon Kim and Sang Il Park. 2018. Deep Context-Aware Descreening
and Rescreening of Halftone Images. ACM Trans. Graph. 37, 4, Article 48
(August 2018), 12 pages. [DOI](https://doi.org/10.1145/3197517.3201377)<br>

## This Implementation
Nikan Doosti. (2021). Nikronic/Deep-Halftoning: v0.1-Zenodo-pre-alpha (v0.1-Zenodo-pre-alpha). Zenodo. https://doi.org/10.5281/zenodo.5651805

[![DOI](https://zenodo.org/badge/424633120.svg)](https://zenodo.org/badge/latestdoi/424633120)
