
//This file is part of the MSD-Detector project (github.com/fedassa/msdDetector).
//
//The MSD-Detector is free software : you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//The MSD-Detector is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with the MSD-Detector project.If not, see <http://www.gnu.org/licenses/>.
// 
// AUTHOR: Federico Tombari (fedassa@gmail.com)
// University of Bologna, Open Perception

#ifndef LSD_DETECTOR_H_
#define LSD_DETECTOR_H_

//#include <vector>
//#include "opencv/cv.hpp"

#include "MSDpoint.h"
#include "StdAfx.h"
//#define BOOST_MULTICORE
#ifdef BOOST_MULTICORE
#include "boost\thread.hpp"
#endif
#include <assert.h>

class MsdDetector
{
public:

	MsdDetector()
	{
        mDebug=0;
		m_patch_radius = 3;
		m_search_area_radius = 5;

		m_nms_radius = 5;
		m_nms_scale_radius = 0;

		m_th_saliency = 1.0;
		m_kNN = 4;

		m_scale_factor = 1.25;
		m_n_scales = -1;
		m_compute_orientation = false;
        m_circular_window=false;
        m_RefinedKps=true;

    }

    std::vector<MSDPoint> detect(Tiff_Im &img);
    std::vector<MSDPoint> detect(Im2D_U_INT2 &img);
    // write intermediate result and print blabla in debug mode
    void setDebug(bool aDebug){ mDebug = aDebug; }
	
    void setPatchRadius(int patchRadius){ m_patch_radius = patchRadius; }
    int getPatchRadius(){ return m_patch_radius; }

    void setSearchAreaRadius(int searchAreaRadius){ m_search_area_radius = searchAreaRadius; }
    int getSearchAreaRadius(){ return m_search_area_radius; }

    void setNMSRadius(int nmsRadius){ m_nms_radius = nmsRadius; }
    int getNMSRadius(){ return m_nms_radius; }

    void setNMSScaleRadius(int nmsScaleRadius){ m_nms_scale_radius = nmsScaleRadius; }
    int getNMSScaleRadius(){ return m_nms_scale_radius; }

    void setThSaliency(float thSaliency){ m_th_saliency = thSaliency; }
    float getThSaliency(){ return m_th_saliency; }

    void setKNN(int kNN){ m_kNN = kNN; }
    int getKNN(){ return m_kNN; }

    void setScaleFactor(float scaleFactor){ m_scale_factor = scaleFactor; }
    float getScaleFactor(){ return m_scale_factor; }

    void setNScales(int nScales){ m_n_scales = nScales; }
    int getNScales(){ return m_n_scales; }

    void setComputeOrientation(bool computeOrientation){ m_compute_orientation = computeOrientation; }
    bool getComputeOrientation(){ return m_compute_orientation; }
	//******************************************************************
    void setCircularWindow(bool circularwindow){m_circular_window=circularwindow;}
    bool getCircularWindow() {return m_circular_window;}
    //*****************************************************************
    void setRefinedKP(bool REFINED){m_RefinedKps=REFINED;}
    bool getRefinedKP() {return m_RefinedKps;}
    //*****************************************************************
	
private: 

    bool mDebug;
	int m_patch_radius;
	int m_search_area_radius;

	int m_nms_radius;
	int m_nms_scale_radius;

	float m_th_saliency;
	int m_kNN;

	float m_scale_factor;
	int m_n_scales;
	int m_cur_n_scales;
	bool m_compute_orientation;
	bool m_circular_window;
    bool m_RefinedKps;

    std::vector< Im2D<U_INT2,INT> > m_scaleSpace;
	
	inline float computeAvgDistance(std::vector<float> &minVals, int den)
	{
	    double avg_dist = 0.0;
		for (unsigned int i = 0; i<minVals.size(); i++)
			avg_dist += minVals[i];

		avg_dist /= den;
		return avg_dist;
	}

    void RefineKP( const Pt2di &i_p, Pt2dr &o_p , float * SaliencyMap, int lvl); // Refine keypoints

    void contextualSelfDissimilarity(Im2D<U_INT2,INT> &img, int xmin, int xmax, float *saliency);

    float computeOrientation(Im2D<U_INT2,INT> &img, int x, int y, std::vector<Pt2df> circle);

    void nonMaximaSuppression(std::vector<float *> & saliency, std::vector<MSDPoint> & keypoints);
};

template <class Type,class TyBase>
class ImagePyramid
{
public:


    ImagePyramid( Im2D <Type,TyBase> &im, const int nLevels, const float scaleFactor = 1.6f);
    Im2D<Type, TyBase> resize(Im2D<Type,TyBase> & im, Pt2dr Size);
    ~ImagePyramid();

    const std::vector< Im2D<Type, TyBase> > getImPyr() const { return m_imPyr; }
private:

    std::vector< Im2D<Type, TyBase> > m_imPyr;
    int m_nLevels;
    float m_scaleFactor;
};

#endif
