# Cross-view Awesome

## Table of contents
- [Geo-localization](#Geo-localization)
- [Synthesis](#Synthesis)

## Geo-localization

1.
    ### Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network [(ECCV 2024)](https://link.springer.com/chapter/10.1007/978-3-031-72913-3_5)
  
    **Authors:** Junyan Ye, Zhutao Lv, Weijia Li, Jinhua Yu, Haote Yang, Huaping Zhong, and Conghui He
  
    <details span>
    <summary><b>Abstract</b></summary>
    Cross-view geolocalization identifies the geographic location of street view images by matching them with a georeferenced satellite database. Significant challenges arise due to the drastic appearance and geometry differences between views. In this paper, we propose a new approach for cross-view image geo-localization, i.e., the Panorama-BEV Co-Retrieval Network. Specifically, by utilizing the ground plane assumption and geometric relations, we convert street view panorama images into the BEV view, reducing the gap between street panoramas and satellite imagery. In the existing retrieval of street view panorama images and satellite images, we introduce BEV and satellite image retrieval branches for collaborative retrieval. By retaining the original street view retrieval branch, we overcome the limited perception range issue of BEV representation. Our network enables comprehensive perception of both the global layout and local details around the street view capture locations. Additionally, we introduce CVGlobal, a global cross-view dataset that is closer to real-world scenarios. This dataset adopts a more realistic setup, with street view directions not aligned with satellite images. CVGlobal also includes cross-regional, cross-temporal, and street view to map retrieval tests, enabling a comprehensive evaluation of algorithm performance. Our method excels in multiple tests on common cross-view datasets such as CVUSA, CVACT, VIGOR, and our newly introduced CVGlobal, surpassing the current state-of-the-art approaches. The code and datasets can be found at https://github.com/yejy53/EP-BEV.
    </details>
    
    [Paper](https://arxiv.org/pdf/2408.05475) | [Code](https://github.com/yejy53/EP-BEV) | [arXiv](https://arxiv.org/abs/2408.05475) | [BibTeX](./citations/ye2024cross)
    
    ---

1.
    ### Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation [(ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Deuser_Sample4Geo_Hard_Negative_Sampling_For_Cross-View_Geo-Localisation_ICCV_2023_paper.html)
  
    **Authors:** Fabian Deuser, Konrad Habel, Norbert Oswald
  
    <details span>
    <summary><b>Abstract</b></summary>
    Cross-View Geo-Localisation is still a challenging task where additional modules, specific pre-processing or zooming strategies are necessary to determine accurate positions of images. Since different views have different geometries, pre-processing like polar transformation helps to merge them. However, this results in distorted images which then have to be rectified. Adding hard negatives to the training batch could improve the overall performance but with the default loss functions in geo-localisation it is difficult to include them. In this article, we present a simplified but effective architecture based on contrastive learning with symmetric InfoNCE loss that outperforms current state-of-the-art results. Our framework consists of a narrow training pipeline that eliminates the need of using aggregation modules, avoids further pre-processing steps and even increases the generalisation capability of the model to unknown regions. We introduce two types of sampling strategies for hard negatives. The first explicitly exploits geographically neighboring locations to provide a good starting point. The second leverages the visual similarity between the image embeddings in order to mine hard negative samples. Our work shows excellent performance on common cross-view datasets like CVUSA, CVACT, University-1652 and VIGOR. A comparison between cross-area and same-area settings demonstrate the good generalisation capability of our model.
    </details>

    [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Deuser_Sample4Geo_Hard_Negative_Sampling_For_Cross-View_Geo-Localisation_ICCV_2023_paper.pdf) | [arXiv](https://arxiv.org/abs/2303.11851) | [BibTeX](./citations/deuser2023sample4geo)
    
    ---

1.
    ### Boosting 3-DoF Ground-to-Satellite Camera Localization Accuracy via Geometry-Guided Cross-View Transformer [(ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_Boosting_3-DoF_Ground-to-Satellite_Camera_Localization_Accuracy_via_Geometry-Guided_Cross-View_Transformer_ICCV_2023_paper.html)
  
    **Authors:** Yujiao Shi, Fei Wu, Akhil Perincherry, Ankit Vora, Hongdong Li
  
    <details span>
    <summary><b>Abstract</b></summary>
    Image retrieval-based cross-view localization methods often lead to very coarse camera pose estimation, due to the limited sampling density of the database satellite images. In this paper, we propose a method to increase the accuracy of a ground camera's location and orientation by estimating the relative rotation and translation between the ground-level image and its matched/retrieved satellite image. Our approach designs a geometry-guided cross-view transformer that combines the benefits of conventional geometry and learnable cross-view transformers to map the ground-view observations to an overhead view. Given the synthesized overhead view and observed satellite feature maps, we construct a neural pose optimizer with strong global information embedding ability to estimate the relative rotation between them. After aligning their rotations, we develop an uncertainty-guided spatial correlation to generate a probability map of the vehicle locations, from which the relative translation can be determined. Experimental results demonstrate that our method significantly outperforms the state-of-the-art. Notably, the likelihood of restricting the vehicle lateral pose to be within 1m of its Ground Truth (GT) value on the cross-view KITTI dataset has been improved from 35.54% to 76.44%, and the likelihood of restricting the vehicle orientation to be within 1 degree of its GT value has been improved from 19.64% to 99.10%.
    </details>

    [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Boosting_3-DoF_Ground-to-Satellite_Camera_Localization_Accuracy_via_Geometry-Guided_Cross-View_Transformer_ICCV_2023_paper.pdf) | [Code](https://github.com/YujiaoShi/Boosting3DoFAccuracy) | [BibTeX](./citations/shi2023boosting)
    
    ---

1.
    ###  Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator [(NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/112d8e0c7563de6e3408b49a09b4d8a3-Abstract-Conference.html)
  
    **Authors:** Xiaolong Wang, Runsen Xu, Zhuofan Cui, Zeyu Wan, Yu Zhang
  
    <details span>
    <summary><b>Abstract</b></summary>
    In this paper, we introduce a novel approach to fine-grained cross-view geo-localization. Our method aligns a warped ground image with a corresponding GPS-tagged satellite image covering the same area using homography estimation. We first employ a differentiable spherical transform, adhering to geometric principles, to accurately align the perspective of the ground image with the satellite map. This transformation effectively places ground and aerial images in the same view and on the same plane, reducing the task to an image alignment problem. To address challenges such as occlusion, small overlapping range, and seasonal variations, we propose a robust correlation-aware homography estimator to align similar parts of the transformed ground image with the satellite image. Our method achieves sub-pixel resolution and meter-level GPS accuracy by mapping the center point of the transformed ground image to the satellite image using a homography matrix and determining the orientation of the ground camera using a point above the central axis. Operating at a speed of 30 FPS, our method outperforms state-of-the-art techniques, reducing the mean metric localization error by 21.3% and 32.4% in same-area and cross-area generalization tasks on the VIGOR benchmark, respectively, and by 34.4% on the KITTI benchmark in same-area evaluation.
    </details>

    [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/112d8e0c7563de6e3408b49a09b4d8a3-Paper-Conference.pdf) | [arXiv](https://arxiv.org/abs/2308.16906) | [BibTeX](./citations/wang2023fine)
    
    ---

1.
    ###  Simple, Effective and General: A New Backbone for Cross-view Image Geo-localization [(arXiv 2023)](https://arxiv.org/abs/2302.01572)
  
    **Authors:** Yingying Zhu, Hongji Yang, Yuxin Lu, Qiang Huang
  
    <details span>
    <summary><b>Abstract</b></summary>
    In this work, we aim at an important but less explored problem of a simple yet effective backbone specific for cross-view geo-localization task. Existing methods for cross-view geo-localization tasks are frequently characterized by 1) complicated methodologies, 2) GPU-consuming computations, and 3) a stringent assumption that aerial and ground images are centrally or orientation aligned. To address the above three challenges for cross-view image matching, we propose a new backbone network, named Simple Attention-based Image Geo-localization network (SAIG). The proposed SAIG effectively represents long-range interactions among patches as well as cross-view correspondence with multi-head self-attention layers. The "narrow-deep" architecture of our SAIG improves the feature richness without degradation in performance, while its shallow and effective convolutional stem preserves the locality, eliminating the loss of patchify boundary information. Our SAIG achieves state-of-the-art results on cross-view geo-localization, while being far simpler than previous works. Furthermore, with only 15.9% of the model parameters and half of the output dimension compared to the state-of-the-art, the SAIG adapts well across multiple cross-view datasets without employing any well-designed feature aggregation modules or feature alignment algorithms. In addition, our SAIG attains competitive scores on image retrieval benchmarks, further demonstrating its generalizability. As a backbone network, our SAIG is both easy to follow and computationally lightweight, which is meaningful in practical scenario. Moreover, we propose a simple Spatial-Mixed feature aggregation moDule (SMD) that can mix and project spatial information into a low-dimensional space to generate feature descriptors
    </details>

    [Code](https://github.com/yanghongji2007/SAIG) | [arXiv](https://arxiv.org/abs/2302.01572) | [BibTeX](./citations/zhu2023simple)
    
    ---


1.
    ### TransGeo: Transformer Is All You Need for Cross-View Image Geo-Localization [(CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_TransGeo_Transformer_Is_All_You_Need_for_Cross-View_Image_Geo-Localization_CVPR_2022_paper.html)
  
    **Authors:** Sijie Zhu, Mubarak Shah, Chen Chen
  
    <details span>
    <summary><b>Abstract</b></summary>
    The dominant CNN-based methods for cross-view image geo-localization rely on polar transform and fail to model global correlation. We propose a pure transformer-based approach (TransGeo) to address these limitations from a different perspective. TransGeo takes full advantage of the strengths of transformer related to global information modeling and explicit position information encoding. We further leverage the flexibility of transformer input and propose an attention-guided non-uniform cropping method, so that uninformative image patches are removed with negligible drop on performance to reduce computation cost. The saved computation can be reallocated to increase resolution only for informative patches, resulting in performance improvement with no additional computation cost. This "attend and zoom-in" strategy is highly similar to human behavior when observing images. Remarkably, TransGeo achieves state-of-the-art results on both urban and rural datasets, with significantly less computation cost than CNN-based methods. It does not rely on polar transform and infers faster than CNN-based methods. Code is available at https://github.com/Jeff-Zilence/TransGeo2022.
    </details>

    [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_TransGeo_Transformer_Is_All_You_Need_for_Cross-View_Image_Geo-Localization_CVPR_2022_paper.pdf) | [Code](https://github.com/Jeff-Zilence/TransGeo2022) | [arXiv](https://arxiv.org/abs/2204.00097) | [BibTeX](./citations/zhu2022transgeo)
    
    ---

1.
    ### Accurate 3-DoF Camera Geo-Localization via Ground-to-Satellite Image Matching [(TPAMI 2022)](https://ieeexplore.ieee.org/abstract/document/9826417)
  
    **Authors:** Yujiao Shi, Xin Yu, Liu Liu, Dylan Campbell, Piotr Koniusz, and Hongdong Li
  
    <details span>
    <summary><b>Abstract</b></summary>
    We address the problem of ground-to-satellite image geo-localization, that is, estimating the camera latitude, longitude and orientation (azimuth angle) by matching a query image captured at the ground level against a large-scale database with geotagged satellite images. Our prior arts treat the above task as pure image retrieval by selecting the most similar satellite reference image matching the ground-level query image. However, such an approach often produces coarse location estimates because the geotag of the retrieved satellite image only corresponds to the image center while the ground camera can be located at any point within the image. To further consolidate our prior research finding, we present a novel geometry-aware geo-localization method. Our new method is able to achieve the fine-grained location of a query image, up to pixel size precision of the satellite image, once its coarse location and orientation have been determined. Moreover, we propose a new geometry-aware image retrieval pipeline to improve the coarse localization accuracy. Apart from a polar transform in our conference work, this new pipeline also maps satellite image pixels to the ground-level plane in the ground-view via a geometry-constrained projective transform to emphasize informative regions, such as road structures, for cross-view geo-localization. Extensive quantitative and qualitative experiments demonstrate the effectiveness of our newly proposed framework. We also significantly improve the performance of coarse localization results compared to the state-of-the-art in terms of location recalls.
    </details>

    [Paper](https://arxiv.org/pdf/2203.14148)  | [arXiv](https://arxiv.org/abs/2203.14148) | [BibTeX](./citations/shi2022accurate)
    
    ---


1.
    ### Cross-view Geo-localization with Layer-to-Layer Transformer [(NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/f31b20466ae89669f9741e047487eb37-Abstract.html)
  
    **Authors:** Hongji Yang, Xiufan Lu, Yingying Zhu
  
    <details span>
    <summary><b>Abstract</b></summary>
    In this work, we address the problem of cross-view geo-localization, which estimates the geospatial location of a street view image by matching it with a database of geo-tagged aerial images. The cross-view matching task is extremely challenging due to drastic appearance and geometry differences across views. Unlike existing methods that predominantly fall back on CNN, here we devise a novel layer-to-layer Transformer (L2LTR) that utilizes the properties of self-attention in Transformer to model global dependencies, thus significantly decreasing visual ambiguities in cross-view geo-localization. We also exploit the positional encoding of the Transformer to help the L2LTR understand and correspond geometric configurations between ground and aerial images. Compared to state-of-the-art methods that impose strong assumptions on geometry knowledge, the L2LTR flexibly learns the positional embeddings through the training objective. It hence becomes more practical in many real-world scenarios. Although Transformer is well suited to our task, its vanilla self-attention mechanism independently interacts within image patches in each layer, which overlooks correlations between layers. Instead, this paper proposes a simple yet effective self-cross attention mechanism to improve the quality of learned representations. Self-cross attention models global dependencies between adjacent layers and creates short paths for effective information flow. As a result, the proposed self-cross attention leads to more stable training, improves the generalization ability, and prevents the learned intermediate features from being overly similar. Extensive experiments demonstrate that our L2LTR performs favorably against state-of-the-art methods on standard, fine-grained, and cross-dataset cross-view geo-localization tasks. 
    </details>

    [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/f31b20466ae89669f9741e047487eb37-Paper.pdf) | [Code](https://github.com/yanghongji2007/cross_view_localization_L2LTR)  | [BibTeX](./citations/yang2021cross)
    
    ---


1.
    ### Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization [(CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Toker_Coming_Down_to_Earth_Satellite-to-Street_View_Synthesis_for_Geo-Localization_CVPR_2021_paper.html)
  
    **Authors:** Aysim Toker, Qunjie Zhou, Maxim Maximov, Laura Leal-Taixe
  
    <details span>
    <summary><b>Abstract</b></summary>
    The goal of cross-view image based geo-localization is to determine the location of a given street view image by matching it against a collection of geo-tagged satellite images. This task is notoriously challenging due to the drastic viewpoint and appearance differences between the two domains. We show that we can address this discrepancy explicitly by learning to synthesize realistic street views from satellite inputs. Following this observation, we propose a novel multi-task architecture in which image synthesis and retrieval are considered jointly. The rationale behind this is that we can bias our network to learn latent feature representations that are useful for retrieval if we utilize them to generate images across the two input domains. To the best of our knowledge, ours is the first approach that creates realistic street views from satellite images and localizes the corresponding query street view simultaneously in an end-to-end manner. In our experiments, we obtain state-of-the-art performance on the CVUSA and CVACT benchmarks. Finally, we show compelling qualitative results for satellite-to-street view synthesis.
    </details>

    [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Toker_Coming_Down_to_Earth_Satellite-to-Street_View_Synthesis_for_Geo-Localization_CVPR_2021_paper.pdf) | [arXiv](https://arxiv.org/abs/2103.06818) | [BibTeX](./citations/toker2021coming)
    
    ---

1.
    ###  VIGOR: Cross-View Image Geo-Localization Beyond One-to-One Retrieval [(CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_VIGOR_Cross-View_Image_Geo-Localization_Beyond_One-to-One_Retrieval_CVPR_2021_paper.html)
  
    **Authors:** Sijie Zhu, Taojiannan Yang, Chen Chen
  
    <details span>
    <summary><b>Abstract</b></summary>
    Cross-view image geo-localization aims to determine the locations of street-view query images by matching with GPS-tagged reference images from aerial view. Recent works have achieved surprisingly high retrieval accuracy on city-scale datasets. However, these results rely on the assumption that there exists a reference image exactly centered at the location of any query image, which is not applicable for practical scenarios. In this paper, we redefine this problem with a more realistic assumption that the query image can be arbitrary in the area of interest and the reference images are captured before the queries emerge. This assumption breaks the one-to-one retrieval setting of existing datasets as the queries and reference images are not perfectly aligned pairs, and there may be multiple reference images covering one query location. To bridge the gap between this realistic setting and existing datasets, we propose a new large-scale benchmark --VIGOR-- for cross-View Image Geo-localization beyond One-to-one Retrieval. We benchmark existing state-of-the-art methods and propose a novel end-to-end framework to localize the query in a coarse-to-fine manner. Apart from the image-level retrieval accuracy, we also evaluate the localization accuracy in terms of the actual distance (meters) using the raw GPS data. Extensive experiments are conducted under different application scenarios to validate the effectiveness of the proposed method. The results indicate that cross-view geo-localization in this realistic setting is still challenging, fostering new research in this direction. Our dataset and code will be publicly available.
    </details>

    [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_VIGOR_Cross-View_Image_Geo-Localization_Beyond_One-to-One_Retrieval_CVPR_2021_paper.pdf) | [Code](https://github.com/JeffZilence/VIGOR.
) | [arXiv](https://arxiv.org/abs/2011.12172) | [BibTeX](./citations/zhu2021vigor)
    
    ---


1.
    ###  University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization[(MM 2020)](https://dl.acm.org/doi/abs/10.1145/3394171.3413896)
  
    **Authors:** Zhedong Zheng, Yunchao Wei, Yi Yang
  
    <details span>
    <summary><b>Abstract</b></summary>
    We consider the problem of cross-view geo-localization. The primary challenge is to learn the robust feature against large viewpoint changes. Existing benchmarks can help, but are limited in the number of viewpoints. Image pairs, containing two viewpoints, e.g., satellite and ground, are usually provided, which may compromise the feature learning. Besides phone cameras and satellites, in this paper, we argue that drones could serve as the third platform to deal with the geo-localization problem. In contrast to traditional ground-view images, drone-view images meet fewer obstacles, e.g., trees, and provide a comprehensive view when flying around the target place. To verify the effectiveness of the drone platform, we introduce a new multi-view multi-source benchmark for drone-based geo-localization, named University-1652. University-1652 contains data from three platforms, i.e., synthetic drones, satellites and ground cameras of 1,652 university buildings around the world. To our knowledge, University-1652 is the first drone-based geo-localization dataset and enables two new tasks, i.e., drone-view target localization and drone navigation. As the name implies, drone-view target localization intends to predict the location of the target place via drone-view images. On the other hand, given a satellite-view query image, drone navigation is to drive the drone to the area of interest in the query. We use this dataset to analyze a variety of off-the-shelf CNN features and propose a strong CNN baseline on this challenging dataset. The experiments show that University-1652 helps the model to learn viewpoint-invariant features and also has good generalization ability in real-world scenarios.
    </details>

    [Paper](https://arxiv.org/abs/2002.12186) | [arXiv](https://arxiv.org/pdf/2002.12186) | [BibTeX](./citations/zheng2020university)
    
    ---


1.
    ### Spatial-Aware Feature Aggregation for Image based Cross-View Geo-Localization [(NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/ba2f0015122a5955f8b3a50240fb91b2-Abstract.html)
  
    **Authors:** Yujiao Shi, Liu Liu, Xin Yu, Hongdong Li
  
    <details span>
    <summary><b>Abstract</b></summary>
    In this paper, we develop a new deep network to explicitly address these inherent differences between ground and aerial views. We observe there exist some approximate domain correspondences between ground and aerial images. Specifically, pixels lying on the same azimuth direction in an aerial image approximately correspond to a vertical image column in the ground view image. Thus, we propose a two-step approach to exploit this prior knowledge. The first step is to apply a regular polar transform to warp an aerial image such that its domain is closer to that of a ground-view panorama. Note that polar transform as a pure geometric transformation is agnostic to scene content, hence cannot bring the two domains into full alignment. Then, we add a subsequent spatial-attention mechanism which further brings corresponding deep features closer in the embedding space. To improve the robustness of feature representation, we introduce a feature aggregation strategy via learning multiple spatial embeddings. By the above two-step approach, we achieve more discriminative deep representations, facilitating cross-view Geo-localization more accurate. Our experiments on standard benchmark datasets show significant performance boosting, achieving more than doubled recall rate compared with the previous state of the art.
    </details>

    [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/ba2f0015122a5955f8b3a50240fb91b2-Paper.pdf) | [Code](https://github.com/YujiaoShi/cross_view_localization_SAFA) | [BibTeX](./citations/shi2019spatial)
    
    ---

1.
    ###  Lending Orientation to Neural Networks for Cross-View Geo-Localization [(CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Lending_Orientation_to_Neural_Networks_for_Cross-View_Geo-Localization_CVPR_2019_paper.html)
  
    **Authors:** Liu Liu, Hongdong Li
  
    <details span>
    <summary><b>Abstract</b></summary>
    This paper studies image-based geo-localization (IBL) problem using ground-to-aerial cross-view matching. The goal is to predict the spatial location of a ground-level query image by matching it to a large geotagged aerial image database (e.g., satellite imagery). This is a challenging task due to the drastic differences in their viewpoints and visual appearances. Existing deep learning methods for this problem have been focused on maximizing feature similarity between spatially close-by image pairs, while minimizing other images pairs which are far apart. They do so by deep feature embedding based on visual appearance in those ground-and-aerial images. However, in everyday life, humans commonly use orientation information as an important cue for the task of spatial localization. Inspired by this insight, this paper proposes a novel method which endows deep neural networks with the `commonsense' of orientation. Given a ground-level spherical panoramic image as query input (and a large georeferenced satellite image database), we design a Siamese network which explicitly encodes the orientation (i.e., spherical directions) of each pixel of the images. Our method significantly boosts the discriminative power of the learned deep features, leading to a much higher recall and precision outperforming all previous methods. Our network is also more compact using only 1/5th number of parameters than a previously best-performing network. To evaluate the generalization of our method, we also created a large-scale cross-view localization benchmark containing 100K geotagged ground-aerial pairs covering a city. Our codes and datasets are available at https://github.com/Liumouliu/OriCNN.
    </details>

    [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Lending_Orientation_to_Neural_Networks_for_Cross-View_Geo-Localization_CVPR_2019_paper.pdf) | [Code](https://github.com/Liumouliu/OriCNN.
) | [arXiv](https://arxiv.org/abs/1903.12351) | [BibTeX](./citations/liu2019lending)
    
    ---


1.
    ### Cross-View Image Matching for Geo-Localization in Urban Environments [(CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Tian_Cross-View_Image_Matching_CVPR_2017_paper.html)
  
    **Authors:** Yicong Tian, Chen Chen, Mubarak Shah
  
    <details span>
    <summary><b>Abstract</b></summary>
    In this paper, we address the problem of cross-view image geo-localization. Specifically, we aim to estimate the GPS location of a query street view image by finding the matching images in a reference database of geo-tagged bird's eye view images, or vice versa. To this end, we present a new framework for cross-view image geo-localization by taking advantage of the tremendous success of deep convolutional neural networks (CNNs) in image classification and object detection. First, we employ the Faster R-CNN to detect buildings in the query and reference images. Next, for each building in the query image, we retrieve the k nearest neighbors from the reference buildings using a Siamese network trained on both positive matching image pairs and negative pairs. To find the correct NN for each query building, we develop an efficient multiple nearest neighbors matching method based on dominant sets. We evaluate the proposed framework on a new dataset that consists of pairs of street view and bird's eye view images. Experimental results show that the proposed method achieves better geo-localization accuracy than other approaches and is able to generalize to images at unseen locations.
    </details>

    [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tian_Cross-View_Image_Matching_CVPR_2017_paper.pdf) | [arXiv](https://arxiv.org/abs/1703.07815) | [BibTeX](./citations/tian2017cross)
    
    ---

1. 
    ### Localizing and Orienting Street Views Using Overhead Imagery [(ECCV 2016)](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_30)

    **Authors:** Nam N. Vo, James Hays

    <details span>
    <summary><b>Abstract</b></summary>
    In this paper we aim to determine the location and orientation of a ground-level query image by matching to a reference database of overhead (e.g. satellite) images. For this task we collect a new dataset with one million pairs of street view and overhead images sampled from eleven U.S. cities. We explore several deep CNN architectures for cross-domain matching – Classification, Hybrid, Siamese, and Triplet networks. Classification and Hybrid architectures are accurate but slow since they allow only partial feature precomputation. We propose a new loss function which significantly improves the accuracy of Siamese and Triplet embedding networks while maintaining their applicability to large-scale retrieval tasks like image geolocalization. This image matching task is challenging not just because of the dramatic viewpoint difference between ground-level and overhead imagery but because the orientation (i.e. azimuth) of the street views is unknown making correspondence even more difficult. We examine several mechanisms to match in spite of this – training for rotation invariance, sampling possible rotations at query time, and explicitly predicting relative rotation of ground and overhead images with our deep networks. It turns out that explicit orientation supervision also improves location prediction accuracy. Our best performing architectures are roughly 2.5 times as accurate as the commonly used Siamese network baseline.
    </details>

    [Paper](https://link.springer.com/content/pdf/10.1007/978-3-319-46448-0_30.pdf?pdf=inline%20link) | [arXiv](https://arxiv.org/abs/1608.00161) | [BibTeX](./citations/vo2016localizing)

    ---

1.
    ### Wide-Area Image Geolocalization With Aerial Reference Imagery [(ICCV 2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Workman_Wide-Area_Image_Geolocalization_ICCV_2015_paper.html)

    **Authors:** Scott Workman, Richard Souvenir, Nathan Jacobs
   
    <details span>
    <summary><b>Abstract</b></summary>
    We propose to use deep convolutional neural networks to address the problem of cross-view image geolocalization, in which the geolocation of a ground-level query image is estimated by matching to georeferenced aerial images. We use state-of-the-art feature representations for ground-level images and introduce a cross-view training approach for learning a joint semantic feature representation for aerial images. We also propose a network architecture that fuses features extracted from aerial images at multiple spatial scales. To support training these networks, we introduce a massive database that contains pairs of aerial and ground-level images from across the United States. Our methods significantly out-perform the state of the art on two benchmark datasets. We also show, qualitatively, that the proposed feature representations are discriminative at both local and continental spatial scales.
    </details>

    [Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Workman_Wide-Area_Image_Geolocalization_ICCV_2015_paper.pdf) | [arXiv](https://arxiv.org/abs/1510.03743) | [BibTeX](./citations/workman2015wide)

    ---

1.
    ### Cross-View Image Geolocalization [(CVPR 2013)](https://openaccess.thecvf.com/content_cvpr_2013/html/Lin_Cross-View_Image_Geolocalization_2013_CVPR_paper.html)
  
    **Authors:** Tsung-Yi Lin, Serge Belongie, James Hays
  
    <details span>
    <summary><b>Abstract</b></summary>
    The recent availability of large amounts of geotagged imagery has inspired a number of data driven solutions to the image geolocalization problem. Existing approaches predict the location of a query image by matching it to a database of georeferenced photographs. While there are many geotagged images available on photo sharing and street view sites, most are clustered around landmarks and urban areas. The vast majority of the Earth's land area has no ground level reference photos available, which limits the applicability of all existing image geolocalization methods. On the other hand, there is no shortage of visual and geographic data that densely covers the Earth we examine overhead imagery and land cover survey data but the relationship between this data and ground level query photographs is complex. In this paper, we introduce a cross-view feature translation approach to greatly extend the reach of image geolocalization methods. We can often localize a query even if it has no corresponding groundlevel images in the database. A key idea is to learn the relationship between ground level appearance and overhead appearance and land cover attributes from sparsely available geotagged ground-level images. We perform experiments over a 1600 km d-region containing a variety of scenes and land cover types. For each query, our algorithm produces a probability density over the region of interest.
    </details>

    [Paper](https://openaccess.thecvf.com/content_cvpr_2013/papers/Lin_Cross-View_Image_Geolocalization_2013_CVPR_paper.pdf) | [BibTeX](./citations/lin2013cross)

    ---



<p>&nbsp;</p>



## Synthesis
 
1.
    ### Crossviewdiff: A cross-view diffusion model for satellite-to-street view synthesis
  
    **Authors:** Weijia Li, Jun He, Junyan Ye, Huaping Zhong, Zhimeng Zheng, Zilong Huang, Dahua Lin, Conghui He
  
    <details span>
    <summary><b>Abstract</b></summary>
    Satellite-to-street view synthesis aims at generating a realistic street-view image from its corresponding satellite-view image. Although stable diffusion models have exhibit remarkable performance in a variety of image generation applications, their reliance on similar-view inputs to control the generated structure or texture restricts their application to the challenging cross-view synthesis task. In this work, we propose CrossViewDiff, a cross-view diffusion model for satellite-to-street view synthesis. To address the challenges posed by the large discrepancy across views, we design the satellite scene structure estimation and cross-view texture mapping modules to construct the structural and textural controls for street-view image synthesis. We further design a cross-view control guided denoising process that incorporates the above controls via an enhanced cross-view attention module. To achieve a more comprehensive evaluation of the synthesis results, we additionally design a GPT-based scoring method as a supplement to standard evaluation metrics. We also explore the effect of different data sources (e.g., text, maps, building heights, and multi-temporal satellite imagery) on this task. Results on three public cross-view datasets show that CrossViewDiff outperforms current state-of-the-art on both standard and GPT-based evaluation metrics, generating high-quality street-view panoramas with more realistic structures and textures across rural, suburban, and urban scenes.
    </details>

    [Paper](https://arxiv.org/pdf/2408.14765) | [Code](https://opendatalab.github.io/CrossViewDiff/) | [BibTeX](./citations/li2024crossviewdiff.txt)
  
    ---
