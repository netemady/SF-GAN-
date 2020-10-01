# wb_command -gifti-label-to-roi 100307.L.aparc.32k_fs_LR.label.gii \ ../ROIs/LPrecen.func.gii -name L_precentral
# wb_command -gifti-label-to-roi 100307.L.aparc.32k_fs_LR.label.gii \ ../ROIs/LOccip.func.gii -name L_lateraloccipital
# We will also convert the obtained GIFTI surface files to ASCII as before with these 2 calls:
# surf2surf -i 100307.L.white.32k_fs_LR.surf.gii -o \
# ../ROIs/LPrecen.asc --values=../ROIs/LPrecen.func.gii --outputtype=ASCII
# surf2surf -i 100307.L.white.32k_fs_LR.surf.gii -o \
# ../ROIs/LOccip.asc --values=../ROIs/LOccip.func.gii --outputtype=ASCII
# We can now run tractography using the surface ROIs and the thalamic volume ROI and obtain a Matrix1 parcellated Connectome.
# IF you had sufficient computer memory, your probtrackx call would be (don’t run this):
# cd /data/practicals/day3/Diffusion/100307/MNINonLinear echo "ROIs/LPrecen.asc" > seeds.txt 
# echo "ROIs/LOccip.asc" >> seeds.txt
# echo "ROIs/THALAMUS_LEFT.nii.gz" >> seeds.txt
# probtrackx2 --samples=../T1w/Diffusion.bedpostX/merged \ --mask=../T1w/Diffusion.bedpostX/nodif_brain_mask --seed=seeds.txt \ --xfm=xfms/standard2acpc_dc --invxfm=xfms/acpc_dc2standard \ --seedref=T1w_restore.2.nii.gz --loopcheck --forcedir --network --omatrix1 \
# --nsamples=20 --avoid=ROIs/CSFmask.nii.gz -V 1 --dir=Network

subject = '100307'
path = '/Users/qz/sampleHCPdataset/'
f = open('sc.sh', 'w')
f.write('cd '+path+subject+'/MNINonLinear/fsaverage_LR32k\n')
for i in range(1,36):
    if i==4: continue;
    f.write('wb_command -gifti-label-to-roi 100307.L.aparc.32k_fs_LR.label.gii ../ROIs/L'+str(i)+'.func.gii -key '+ str(i)+'\n')
    # f.write('wb_command -gifti-label-to-roi 100307.R.aparc.32k_fs_LR.label.gii ../ROIs/R'+str(i)+'.func.gii -key '+ str(i)+'\n')
    f.write(f'surf2surf -i 100307.L.white.32k_fs_LR.surf.gii -o ../ROIs/L{i}.asc --values=../ROIs/L{i}.func.gii --outputtype=ASCII\n')
    # f.write(f'surf2surf -i 100307.R.white.32k_fs_LR.surf.gii -o ../ROIs/R{i}.asc --values=../ROIs/R{i}.func.gii --outputtype=ASCII\n')

for i in range(1,36):
    if i==4: continue;
    # f.write('wb_command -gifti-label-to-roi 100307.L.aparc.32k_fs_LR.label.gii ../ROIs/L'+str(i)+'.func.gii -key '+ str(i)+'\n')
    f.write('wb_command -gifti-label-to-roi 100307.R.aparc.32k_fs_LR.label.gii ../ROIs/R'+str(i)+'.func.gii -key '+ str(i)+'\n')
    f.write(f'surf2surf -i 100307.R.white.32k_fs_LR.surf.gii -o ../ROIs/R{i}.asc --values=../ROIs/R{i}.func.gii --outputtype=ASCII\n')
f.write('cd ..\n')
for i in range(1,36):
    if i==4: continue;
    if i == 1:
        f.write('echo \"ROIs/L1.asc\" > seeds.txt\n')
    else:
        f.write(f'echo \"ROIs/L{i}.asc\" >> seeds.txt\n')

for i in range(1,36):
    if i==4: continue;
    f.write(f'echo \"ROIs/R{i}.asc\" >> seeds.txt\n')
f.write('probtrackx2_gpu --samples=../T1w/Diffusion.bedpostX/merged --mask=../T1w/Diffusion.bedpostX/nodif_brain_mask --seed=seeds.txt --xfm=xfms/standard2acpc_dc --invxfm=xfms/acpc_dc2standard --seedref=T1w_restore.2.nii.gz --loopcheck --forcedir --network --omatrix1 --nsamples=20 -V 1 --dir=Network')
f.close()

# probtrackx2 --samples=../T1w/Diffusion.bedpostX/merged --mask=../T1w/Diffusion.bedpostX/nodif_brain_mask --seed=seeds.txt --xfm=xfms/standard2acpc_dc --invxfm=xfms/acpc_dc2standard --seedref=T1w_restore.2.nii.gz --loopcheck --forcedir --network --omatrix1 --nsamples=20 -V 1 --dir=Network