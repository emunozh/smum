#Generate UML diagrams

root_path="../smum/"

for module in "microsim/" "urbanmet/"
  do
  for f in $root_path$module*.py
  do
  if `echo ${f} | grep "__init__" 1>/dev/null 2>&1`
  then
    echo "Skip init"
  else
    file=${f##*/}
    filename="${file%.*}"
    echo "Generating UML for --> $f"
    pyreverse -Amy -k -o png $f -p $filename
    mv classes_$filename.png _static/images/classes_$filename.png
    pyreverse -Amy -o png $f -p $filename
    mv classes_$filename.png _static/images/classes_M_$filename.png
  fi
  done
done
