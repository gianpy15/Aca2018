#!/usr/bin/env bash
cd test_imgs
find . -depth -name '*.JPEG*' -execdir bash -c 'mv -i "$1" "${1//.JPEG/.jpg}"' bash {} \;
