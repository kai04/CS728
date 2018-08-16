
data_dir=data

for file in $data_dir/*;do
	filename=${file##*/}
	echo $filename,$file
	#python2 preprocess.py $file $filename
	python2 2.py $file $filename
	break
	done
