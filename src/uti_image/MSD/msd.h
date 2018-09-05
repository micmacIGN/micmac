
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
#include "../../uti_image/Digeo/Digeo.h"
#include "DescriptorExtractor.h"
#include "StdAfx.h"
//#define BOOST_MULTICORE
#ifdef BOOST_MULTICORE
#include "boost\thread.hpp"
#endif
#include <assert.h>

class cPtsCaracMSD;

template <class tData, class tComp>

extern void gradient_sob(Im2D<tData,tComp> &i_image, REAL8 i_maxValue, Im2D<REAL4,REAL8> &o_gradient );

extern void Migrate2Lab2wallis(Tiff_Im &image, Im2D<U_INT1,INT> &Output);
extern void Migrate2Lab2wallis(Im2D<U_INT1,INT> &image, Im2D<U_INT1,INT> &Output);
extern void wallis(Im2D<U_INT1,INT> &image, Im2D<U_INT1,INT> &WallEqIm);
template <class Type, class TyBase>
extern void Resizeim(Im2D<Type,TyBase> & im, Im2D<Type,TyBase> & Out, Pt2dr Newsize);

class MsdDetector
{
public:

	MsdDetector()
	{
        mNameIm="MSD";
        mDebug=0;
        mTmpDir="Tmp-MM-Dir";
		m_patch_radius = 3;
		m_search_area_radius = 5;
        //Non-Maxima Suppression
		m_nms_radius = 5;
		m_nms_scale_radius = 0;
		m_th_saliency = 1.0;
		m_kNN = 4;
		m_scale_factor = 1.25;
		m_n_scales = -1;
        //m_compute_orientation = false;
        m_circular_window=false;
        m_RefinedKps=true;
    }

    void saliency2Im2D(const std::vector<float *> &  saliency);

    template <class Type, class TyBase>
    void detect(Im2D<Type,TyBase> &img);

    template <class Type, class TyBase>
    void detect2(Im2D<Type,TyBase> &img);

    void writeKp(std::string aOut){
        DigeoPoint::writeDigeoFile(aOut,mVDP);
    }

    // getter and setter
    const std::vector<DigeoPoint> Kps(){ return mVDP;}
    void setDebug(bool aDebug){ mDebug = aDebug;}
    void setNameIm(std::string aName){ mNameIm = aName;}
    void setDir(std::string aDir){ mTmpDir = aDir; }
    const std::string Dir(){return mTmpDir; }
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

    /* not used in micmac
    void setCircularWindow(bool circularwindow){m_circular_window=circularwindow;}
    bool getCircularWindow() {return m_circular_window;}
    void setRefinedKP(bool REFINED){m_RefinedKps=REFINED;}
    bool getRefinedKP() {return m_RefinedKps;}
    */

    void orientate(Im2D<float, double> &img, std::vector<cPtsCaracMSD> &aVPCar);
    void orientationAndDescriptor();
	
private: 

    bool mDebug;
    std::string mTmpDir,mNameIm;
	int m_patch_radius;
	int m_search_area_radius;
	int m_nms_radius;
    int m_nms_scale_radius;// not used
	float m_th_saliency;
	int m_kNN;
	float m_scale_factor;
	int m_n_scales;
	int m_cur_n_scales;
    //bool m_compute_orientation;
	bool m_circular_window;
    bool m_RefinedKps;

    std::vector< Im2D<U_INT1,INT> > m_scaleSpace;
    std::vector< Im2D<REAL4,REAL8> > mSaliencyIm;
    // keypoint: pt detected with MSD approach. may be filtered afteward
    std::vector<std::vector<cPtsCaracMSD>*> mVVKP;
    // Digeo points: pt with orientation and descriptor, ready to used in micmac pipeline

    std::vector<DigeoPoint> mVDP;
	
	inline float computeAvgDistance(std::vector<float> &minVals, int den)
	{
	    double avg_dist = 0.0;
		for (unsigned int i = 0; i<minVals.size(); i++)
			avg_dist += minVals[i];

		avg_dist /= den;
		return avg_dist;
	}

    //void RefineKP( const Pt2di &i_p, Pt2dr &o_p , float * SaliencyMap, int lvl); // Refine keypoints
    //float computeOrientation(Im2D<U_INT1,INT> &img, int x, int y, std::vector<Pt2df> circle);

    void contextualSelfDissimilarity(Im2D<U_INT1,INT> &img, int xmin, int xmax, float *saliency);
    void nonMaximaSuppression();
    // this is an implementation of method used in Digeo
    int orientate( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracMSD &i_p, REAL8 o_angles[DIGEO_MAX_NB_ANGLES] );


    template <class Type, class TyBase>
    void doIllu(Im2D<Type, TyBase> &img);

    void computeMSDPyram();
    void selectCorners();
    void writeSaliency();

    //template <class tData, class tComp>
    //DigeoPoint ToDigeo(cPtsCaracMSD & aPt,DescriptorExtractor<REAL4,REAL8> & aDesc);
};

class cPtsCaracMSD
{
    public :
       cPtsCaracMSD(){
           addAngle(0.0);
           mMulSc=0;
       }

       Pt2dr         mPt;
       double           mScale;
       double        mSize;
       Pt2dr         mPtSc; // Pt in scaled images
       bool          mMulSc; // detected at least at two scales; may be usefull for some filtering
       std::vector<float> m_angle;

       void setAngle(float angle, unsigned int index){
           if (index<m_angle.size()) {m_angle.at(index)=angle;}
           else {std::cout << "cPtsCaractMSD: index " << index << " for angle " << angle << "is out of range\n";}}
       void addAngle(float angle){m_angle.push_back(angle);}
       float getAngle(unsigned int index){return m_angle.at(index);}
       float getAngle(){return m_angle.at(0);}
       std::vector<float> getAngles(){return m_angle;}
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
