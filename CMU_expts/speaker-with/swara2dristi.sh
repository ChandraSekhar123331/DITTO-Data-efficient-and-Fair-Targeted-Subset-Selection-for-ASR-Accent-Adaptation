for file in */m*/T*t/a*/*/*/*I/*/*/*/r*/t*/t*; do echo $file ${file/train./dristi_train.}; done
for file in */m*/T*t/w*/*/*/*/r*/t*/t*; do echo $file ${file/train./dristi_train.}; done
for file in */m*/T*t/a*/*/*/*/r*/t*/t*; do echo $file ${file/train./dristi_train.}; done

for file in */m*/T*t/a*/*/*/*I/*/*/*/r*/t*/d*; do sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' $file; done
for file in */m*/T*t/w*/*/*/*/r*/t*/d*; do sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' $file; done
for file in */m*/T*t/a*/*/*/*/r*/t*/d*; do sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' $file; done
