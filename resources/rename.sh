find . -depth -name '*.JPEG*' -execdir bash -c 'mv -i "$1" "${1//.JPEG/.jpg}"' bash {} \;
