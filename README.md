# awesome-Cross-view

## Table of contents
- [Geo-localization](#Geo-localization)
- [Synthesis](#Synthesis)

## Geo-localization

1.
    ### Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network [(ECCV 2024)](https://link.springer.com/chapter/10.1007/978-3-031-72913-3_5)
  
    **Authors:** Junyan Ye, Zhutao Lv, Weijia Li, Jinhua Yu, Haote Yang, Huaping Zhong, and Conghui He
  
    <details span>
    <summary><b>Abstract</b></summary>
    Cross-view geolocalization identifies the geographic location of street view images by matching them with a georeferenced satellite database. Significant challenges arise due to the drastic appearance and geometry differences between views. In this paper, we propose a new approach for cross-view image geo-localization, i.e., the Panorama-BEV Co-Retrieval Network. Specifically, by utilizing the ground plane assumption and geometric relations, we convert street view panorama images into the BEV view, reducing the gap between street panoramas and satellite imagery. In the existing retrieval of street view panorama images and satellite images, we introduce BEV and satellite image retrieval branches for collaborative retrieval. By retaining the original street view retrieval branch, we overcome the limited perception range issue of BEV representation. Our network enables comprehensive perception of both the global layout and local details around the street view capture locations. Additionally, we introduce CVGlobal, a global cross-view dataset that is closer to real-world scenarios. This dataset adopts a more realistic setup, with street view directions not aligned with satellite images. CVGlobal also includes cross-regional, cross-temporal, and street view to map retrieval tests, enabling a comprehensive evaluation of algorithm performance. Our method excels in multiple tests on common cross-view datasets such as CVUSA, CVACT, VIGOR, and our newly introduced CVGlobal, surpassing the current state-of-the-art approaches.
    </details>
    
    [Paper](https://arxiv.org/pdf/2408.05475) | [Code](https://github.com/yejy53/EP-BEV) | [arXiv](https://arxiv.org/abs/2408.05475) | [BibTeX](./citations/ye2024cross)
    
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

    **Authors:** Nam N. Vo & James Hays

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

    [Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Workman_Wide-Area_Image_Geolocalization_ICCV_2015_paper.pdf) | [arXiv](https://arxiv.org/abs/1510.03743) | [BibTeX](./citations/Workman_2015_ICCV)

    ---

1.
    ### Cross-View Image Geolocalization [(CVPR 2013)](https://openaccess.thecvf.com/content_cvpr_2013/html/Lin_Cross-View_Image_Geolocalization_2013_CVPR_paper.html)
  
    **Authors:** Tsung-Yi Lin, Serge Belongie, James Hays
  
    <details span>
    <summary><b>Abstract</b></summary>
    The recent availability of large amounts of geotagged imagery has inspired a number of data driven solutions to the image geolocalization problem. Existing approaches predict the location of a query image by matching it to a database of georeferenced photographs. While there are many geotagged images available on photo sharing and street view sites, most are clustered around landmarks and urban areas. The vast majority of the Earth's land area has no ground level reference photos available, which limits the applicability of all existing image geolocalization methods. On the other hand, there is no shortage of visual and geographic data that densely covers the Earth we examine overhead imagery and land cover survey data but the relationship between this data and ground level query photographs is complex. In this paper, we introduce a cross-view feature translation approach to greatly extend the reach of image geolocalization methods. We can often localize a query even if it has no corresponding groundlevel images in the database. A key idea is to learn the relationship between ground level appearance and overhead appearance and land cover attributes from sparsely available geotagged ground-level images. We perform experiments over a 1600 km d-region containing a variety of scenes and land cover types. For each query, our algorithm produces a probability density over the region of interest.
    </details>

    [Paper](https://openaccess.thecvf.com/content_cvpr_2013/papers/Lin_Cross-View_Image_Geolocalization_2013_CVPR_paper.pdf) | [BibTeX](./citations/Lin_2013_CVPR)

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
