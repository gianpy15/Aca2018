//
// Created by gianpaolo on 5/26/18.
//

#include "h5io.h"
using namespace std;
using namespace H5;

const H5std_string FILE_NAME( "../../resources/vgg.h5" );

int main() {
    auto *f = new H5File(FILE_NAME, H5F_ACC_RDONLY);

}