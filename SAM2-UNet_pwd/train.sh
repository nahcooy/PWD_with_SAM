CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "./sam2_hiera_large.pt" \
--train_image_path "D:\pwd\pwd_original_img" \
--train_mask_path "D:\pwd\pwd_gt_img\pwd_gt_img" \
--save_path "D:\sam2unet_save\1009" \
--epoch 50 \
--lr 0.001 \
--batch_size 12