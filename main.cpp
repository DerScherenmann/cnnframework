/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   main.cpp
 * Author: broesel233
 *
 * Created on 5 September 2020, 17:10
 */

#include <iostream>

#include "convnetwork.h"

/*
 *
 */
int main(int argc, char** argv) {

    Convolutional conv = Convolutional({28,28,1},{10,1},{1,2},1,5,4,2,1);
    while(getchar() != 0){
        conv.run_tests();
    }

    return 0;
}

