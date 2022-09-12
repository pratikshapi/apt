cpprun(){
output_file=`echo $1 | cut -d'/' -f 2 | cut -d'.' -f 1`
lab_index=`echo $1 | cut -d'/' -f 1`
mkdir -p builds/$lab_index
g++ -o builds/$lab_index/$output_file "$1"
./builds/$lab_index/$output_file
}
