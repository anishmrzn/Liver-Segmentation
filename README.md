Liver Segmentation

![image](https://github.com/user-attachments/assets/9cffe66e-acfb-4cc8-b31c-1d89bff5b856)
Loss metrics

![image](https://github.com/user-attachments/assets/a68e0f84-3779-4a03-8e02-98545c26fa13)
![image](https://github.com/user-attachments/assets/fd67759c-3686-42c9-94d5-fd18c2027559)
![image](https://github.com/user-attachments/assets/57af84bf-35ca-41bc-a2ec-e27d6da5dfb3)

Output

https://drive.google.com/drive/folders/1E_P51We1UQW3Xpp_RZYBfXakRy-hF5ou?usp=drive_link

drive link for Data and Results

Total Segmentator

![image](https://github.com/user-attachments/assets/cdde7558-7579-44e0-90f1-3af254a6c325)
![image](https://github.com/user-attachments/assets/a03c0b79-a628-4f81-ae7d-224600360b64)

Results for totalsegmentator output

MedSam using totalsegmentator output as prompt box

Processing Volumes:   0%|                                                                                                                              1 [00:00<?, ?it/s]
Processing liver_120.nii.gz with 424 slices...
   Volume liver_120: Dice = 0.8557, Jaccard = 0.7478
Processing Volumes:   9%|███████████▋                                                                                                                     | 1/11 [06:18<1:03:06, 378.60s/it]
Processing liver_121.nii.gz with 463 slices...
   Volume liver_121: Dice = 0.9169, Jaccard = 0.8465
Processing Volumes:  18%|███████████████████████▊                                                                                                           | 2/11 [11:15<49:36, 330.70s/it]
Processing liver_122.nii.gz with 422 slices...
   Volume liver_122: Dice = 0.8655, Jaccard = 0.7629
Processing Volumes:  27%|███████████████████████████████████▋                                                                                               | 3/11 [16:11<41:57, 314.64s/it]
Processing liver_123.nii.gz with 432 slices...
   Volume liver_123: Dice = 0.9162, Jaccard = 0.8453
Processing Volumes:  36%|███████████████████████████████████████████████▋                                                                              
     | 4/11 [21:23<36:35, 313.59s/it]
Processing liver_124.nii.gz with 407 slices...
   Volume liver_124: Dice = 0.9275, Jaccard = 0.8648
Processing Volumes:  45%|███████████████████████████████████████████████████████████▌                                                                  
     | 5/11 [26:23<30:53, 308.90s/it]
Processing liver_125.nii.gz with 410 slices...
   Volume liver_125: Dice = 0.8693, Jaccard = 0.7688
Processing Volumes:  55%|███████████████████████████████████████████████████████████████████████▍                                                      
     | 6/11 [31:33<25:45, 309.17s/it]
Processing liver_126.nii.gz with 401 slices...
   Volume liver_126: Dice = 0.8764, Jaccard = 0.7800
Processing Volumes:  64%|███████████████████████████████████████████████████████████████████████████████████▎                                          
     | 7/11 [36:51<20:47, 311.91s/it]
Processing liver_127.nii.gz with 987 slices...
   Volume liver_127: Dice = 0.8864, Jaccard = 0.7960
Processing Volumes:  73%|███████████████████████████████████████████████████████████████████████████████████████████████▎                              
     | 8/11 [47:31<20:49, 416.55s/it]
Processing liver_128.nii.gz with 654 slices...
   Volume liver_128: Dice = 0.8862, Jaccard = 0.7957
Processing Volumes:  82%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▌                    
   | 9/11 [1:01:10<18:04, 542.28s/it]
Processing liver_129.nii.gz with 338 slices...
Processing liver_129.nii.gz with 338 slices...
   Volume liver_129: Dice = 0.9335, Jaccard = 0.8753
Processing Volumes:  91%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 10/11 [1:13:47<10:08, 608.68s/it]
Processing liver_130.nii.gz with 624 slices...
   Volume liver_130: Dice = 1.0135, Jaccard = 1.0273
Processing Volumes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████Processing Volumes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [1:24:44<00:00, 462.23s/it]

--- Processing Complete ---

Average Dice Score across 11 processed volumes: 0.9043
Average Jaccard Score across 11 processed volumes: 0.8282

Results for medsam output.

MedSam using unet model output as prompt box

--- Starting UNet Inference and MedSAM Integration ---
Processing Volumes (UNet + MedSAM):   0%|                                                                                                                            | 0/11 [00:00<?, ?it/s] 
Processing liver_120.nii.gz with 424 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.8796029724062906].
   Volume liver_120: UNet Dice = 0.8592, UNet Jaccard = 0.7531 | MedSAM Dice = 0.8027, MedSAM Jaccard = 0.6704
Processing Volumes (UNet + MedSAM):   9%|██████████▎                                                                                                      | 1/11 [11:08<1:51:29, 668.92s/it]
Processing liver_121.nii.gz with 463 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.592612357219051].
   Volume liver_121: UNet Dice = 0.7732, UNet Jaccard = 0.6302 | MedSAM Dice = 0.7970, MedSAM Jaccard = 0.6625
Processing Volumes (UNet + MedSAM):  18%|████████████████████▎                                                                                           | 2/11 [33:34<2:40:00, 1066.69s/it]
Processing liver_122.nii.gz with 422 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.2].
   Volume liver_122: UNet Dice = 0.9069, UNet Jaccard = 0.8297 | MedSAM Dice = 0.8772, MedSAM Jaccard = 0.7812
Processing Volumes (UNet + MedSAM):  27%|██████████████████████████████▊                                                                                  | 3/11 [46:56<2:06:08, 946.12s/it]
Processing liver_123.nii.gz with 432 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.2].
   Volume liver_123: UNet Dice = 0.8965, UNet Jaccard = 0.8124 | MedSAM Dice = 0.9096, MedSAM Jaccard = 0.8342
Processing Volumes (UNet + MedSAM):  36%|█████████████████████████████████████████                                                                        | 4/11 [56:02<1:31:57, 788.22s/it]
Processing liver_124.nii.gz with 407 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.894106757998383].
   Volume liver_124: UNet Dice = 0.8764, UNet Jaccard = 0.7800 | MedSAM Dice = 0.8957, MedSAM Jaccard = 0.8112
Processing Volumes (UNet + MedSAM):  45%|████████████████████████████████████████████████████▎                                                              | 5/11 [59:40<58:15, 582.61s/it]
Processing liver_125.nii.gz with 410 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.6459607949252457].
   Volume liver_125: UNet Dice = 0.9476, UNet Jaccard = 0.9004 | MedSAM Dice = 0.8794, MedSAM Jaccard = 0.7848
Processing Volumes (UNet + MedSAM):  55%|█████████████████████████████████████████████████████████████▋                                                   | 6/11 [1:05:07<41:17, 495.56s/it]
Processing liver_126.nii.gz with 401 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9476846537914732].
   Volume liver_126: UNet Dice = 0.9191, UNet Jaccard = 0.8503 | MedSAM Dice = 0.8823, MedSAM Jaccard = 0.7895
Processing Volumes (UNet + MedSAM):  64%|███████████████████████████████████████████████████████████████████████▉                                         | 7/11 [1:09:29<27:57, 419.26s/it]
Processing liver_127.nii.gz with 987 slices using UNet and MedSAM...
   Volume liver_127: UNet Dice = 0.9067, UNet Jaccard = 0.8293 | MedSAM Dice = 0.8442, MedSAM Jaccard = 0.7305
Processing Volumes (UNet + MedSAM):  73%|██████████████████████████████████████████████████████████████████████████████████▏                              | 8/11 [1:21:10<25:26, 508.79s/it]
Processing liver_128.nii.gz with 654 slices using UNet and MedSAM...
   Volume liver_128: UNet Dice = 0.9294, UNet Jaccard = 0.8681 | MedSAM Dice = 0.9001, MedSAM Jaccard = 0.8183
Processing Volumes (UNet + MedSAM):  82%|████████████████████████████████████████████████████████████████████████████████████████████▍                    | 9/11 [1:31:11<17:55, 537.57s/it]
Processing liver_129.nii.gz with 338 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.2].
   Volume liver_129: UNet Dice = 0.7869, UNet Jaccard = 0.6486 | MedSAM Dice = 0.9483, MedSAM Jaccard = 0.9017
Processing Volumes (UNet + MedSAM):  91%|█████████████████████████████████████████████████████████████████████████████████████████████████████▊          | 10/11 [1:39:25<08:44, 524.16s/it]
Processing liver_130.nii.gz with 624 slices using UNet and MedSAM...
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.9764705882352942].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.5257425802507294].
   Volume liver_130: UNet Dice = 0.9202, UNet Jaccard = 0.8521 | MedSAM Dice = 1.0162, MedSAM Jaccard = 1.0329
Processing Volumes (UNet + MedSAM): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [1:53:53<00:00, 621.26s/it]

--- Processing Complete ---

Average UNet Dice Score across 11 processed volumes: 0.8838
Average UNet Jaccard Score across 11 processed volumes: 0.7958
Average MedSAM Dice Score (UNet Prompt) across 11 processed volumes: 0.8866
Average MedSAM Jaccard Score (UNet Prompt) across 11 processed volumes: 0.8016
