/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   PoolLayer.h
 * Author: broesel233
 *
 * Created on 6 September 2020, 15:42
 */

#ifndef POOLLAYER_H
#define POOLLAYER_H

#include <algorithm>
#include <tuple>

#include "layer.h"


class PoolLayer : public Layer {
public:
    //TODO add depth
    PoolLayer(size_t t_width,size_t t_height,size_t t_depth,size_t t_stride) : Layer(t_width,t_height,t_depth,Layer::types::POOL), m_stride(t_stride){};
    //PoolLayer(const PoolLayer& orig) : Layer(orig) {};
    virtual ~PoolLayer(){};
    /**
     * Pool inputs (input must multiple of width and height)
     * @param input
     * @return
     */
    size_t pool(std::vector<std::vector<float>> t_input){

        std::vector<std::vector<float>> pane;
        std::vector<std::vector<std::pair<float, std::pair<size_t,size_t>>>> index_pane;
        for(size_t width = 0;width < t_input.size();width+=m_stride){
            std::vector<float> column;
            std::vector<std::pair<float, std::pair<size_t,size_t>>> index_column;
            for(size_t height = 0;height < t_input.size();height+=m_stride){
                //get highest value and add to outputs
                std::vector<float> values;
                std::pair<float, std::pair<size_t,size_t>> index;
                for(size_t x = 0;x < m_width;x++){
                    for(size_t y = 0;y < m_height;y++){
                        values.push_back(t_input[width+x][height+y]);
                        index.first = t_input[width+x][height+y];
                        index.second = std::make_pair(width+x,height+y);
                    }
                }
                std::sort(values.begin(),values.end(),std::greater<float>());
                column.push_back(values[0]);
                index_column.push_back(index);
            }
            pane.push_back(column);
            index_pane.push_back(index_column);
        }
        m_values = pane;
        m_output_prev_index = index_pane;

        return 0;
    }
    std::vector<std::vector<std::pair<float, std::pair<size_t,size_t>>>> get_org_index(){
        return m_output_prev_index;
    }
private:
    //TODO LMAO <- also this works
    std::vector<std::vector<std::pair<float, std::pair<size_t,size_t>>>> m_output_prev_index;
    size_t m_stride;
};

#endif /* POOLLAYER_H */

