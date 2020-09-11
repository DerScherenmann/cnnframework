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
    PoolLayer(size_t t_width,size_t t_height,size_t t_depth) : Layer(t_width,t_height,t_depth,Layer::types::POOL){};
    PoolLayer(const PoolLayer& orig) : Layer(orig) {};
    virtual ~PoolLayer(){};
    /**
     * Pool inputs (input must multiple of width and height)
     * @param input
     * @return
     */
    size_t pool(std::vector<std::vector<std::vector<float>>> t_input){

        m_output.reserve(m_depth);
        for(size_t depth = 0;depth < t_input.size();depth++){
            std::vector<std::vector<float>> pane;
            std::vector<std::vector<std::pair<float, std::tuple<size_t,size_t,size_t>>>> index_pane;
            for(size_t width = 0;width < t_input[0].size();width+=m_width){
                std::vector<float> column;
                std::vector<std::pair<float, std::tuple<size_t,size_t,size_t>>> index_column;
                for(size_t height = 0;height < t_input[0].size();height+=m_height){
                    //get highest value and add to outputs
                    std::vector<float> values;
                    std::pair<float, std::tuple<size_t,size_t,size_t>> index;
                    for(size_t x = 0;x < m_width;x++){
                        for(size_t y = 0;y < m_height;y++){
                            values.push_back(t_input[depth][width+x][height+y]);
                            index.first = t_input[depth][width+x][height+y];
                            index.second = std::make_tuple(depth,width+x,height+y);
                        }
                    }
                    std::sort(values.begin(),values.end(),std::greater<float>());
                    column.push_back(values[0]);
                    index_column.push_back(index);
                }
                pane.push_back(column);
                index_pane.push_back(index_column);
            }
            m_output.push_back(pane);
            m_output_prev_index.push_back(index_pane);
        }

        return 0;
    }
    std::vector<std::vector<std::vector<float>>> get_outputs(){
        return m_output;
    }
    std::vector<std::vector<std::vector<std::pair<float, std::tuple<size_t,size_t,size_t>>>>> get_org_index(){
        return m_output_prev_index;
    }
private:
    //outer vector width inner height
    std::vector<std::vector<std::vector<float>>> m_output;
    //TODO LMAO
    std::vector<std::vector<std::vector<std::pair<float, std::tuple<size_t,size_t,size_t>>>>> m_output_prev_index;
};

#endif /* POOLLAYER_H */

