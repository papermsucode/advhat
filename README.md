# AdvHat: Real-world adversarial attack on ArcFace Face ID system

By Stepan Komkov and Aleksandr Petiushko

This is the code repository for the AdvHat research article. The article is available [here](https://arxiv.org/abs/1908.08705). The video demo is available [here](https://youtu.be/a4iNg0wWBsQ). Code that is used for the article is available right here.

## Abstract

We propose a novel easily reproducible technique to attack the best public Face ID system ArcFace in different shooting conditions. To create an attack, we print the rectangular paper sticker on a common color printer and put it on the hat. The adversarial sticker is prepared with a novel algorithm for off-plane transformations of the image which imitates sticker location on the hat. Such an approach confuses the state-of-the-art public Face ID model LResNet100E-IR, ArcFace@ms1m-refine-v2 and is transferable to other Face ID models.

## The repository

The repository is organized as follows:

* In the Attack directory, you can find code and instructions on how to reproduce an attack for your images.
* In the Demo directory, you can find a demo script which can help you to verify the robustness of the prepared attack to the real-world shooting conditions.

## Built With

* [InsightFace's ArcFace](https://github.com/deepinsight/insightface) - The SOTA public FaceID model
* [Kevin Zakka's STN](https://github.com/kevinzakka/spatial-transformer-network) - Spatial Transformer realization

## Citation

```
@article{komkov2019advhat,
  title={AdvHat: Real-world adversarial attack on ArcFace Face ID system},
  author={Komkov, Stepan and Petiushko, Aleksandr},
  journal={arXiv preprint arXiv:1908.08705},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/papermsucode/advhat/blob/master/LICENSE) file for details.
