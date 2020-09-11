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

#include "convnetwork.cpp"

/*
 *
 */
int main(int argc, char** argv) {

    Convultional conv = Convultional({28,28,1},5,1,5);
    while(getchar() != 0){
        conv.test();
    }

    return 0;
}

