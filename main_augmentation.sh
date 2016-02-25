#!/bin/bash  
originalpath=~/data/UPMC_Food101/images
newpath=~/data/UPMC_Food101_221_augmented
mkdir -p $newpath
cd $newpath
for dataset in 'test'; do
  SECONDS=0
  echo " "
  echo " "
  echo "# $dataset"
  mkdir -p $dataset
  cd $dataset
  for class in `ls $originalpath/$dataset`; do
    echo " "
    echo "## $class"
    mkdir -p $class
    cd $originalpath/$dataset/$class
    for i in *.jpg; do
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/center_$i" -gravity center -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/northwest_$i" -gravity northwest -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/southwest_$i" -gravity southwest -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/northeast_$i" -gravity northeast -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/southeast_$i" -gravity southeast -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/center_flip_$i" -gravity center -flip -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/northwest_flip_$i" -gravity northwest -flip -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/southwest_flip_$i" -gravity southwest -flip -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/northeast_flip_$i" -gravity northeast -flip -crop 221x221+0+0 +repage "$i"
      mogrify -resize "256x256^" -write "$newpath/$dataset/$class/southeast_flip_$i" -gravity southeast -flip -crop 221x221+0+0 +repage "$i"
    done
    echo "> $SECONDS seconds since start"
    cd $newpath/$dataset
  done
  cd $newpath
done

