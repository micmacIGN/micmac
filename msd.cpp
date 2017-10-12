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



#define _USE_MATH_DEFINES
//#include <math.h>
#include "StdAfx.h"
#include "msdImgPyramid.h"
#include "Keypoint.h"
#include "msd.h"
#include <assert.h>

using namespace std;
   // using namespace cv;

// From code Arnaud LeBris
inline bool Gauss22_invert_b( REAL8 *i_m, REAL8 *i_b )
  {
    #define At(i,j)  (i_m[(i)+(j)*3])
    // Gauss elimination
    for(int j = 0 ; j < 2 ; ++j)
    {
        // look for leading pivot
        REAL8 maxa = 0;
        REAL8 maxabsa = 0;
        int maxi = -1;
        int i;
        for( i=j; i<2; i++ )
        {
            REAL8 a    = At(i,j);
            REAL8 absa = fabsf( a );
            if ( absa>maxabsa )
            {
                maxa    = a ;
                maxabsa = absa ;
                maxi    = i ;
            }
        }

        // singular?
        if ( maxabsa<1e-10 )
        {
            i_m[2]  = 0 ;
            i_m[5]  = 0 ;
            return false;
        }

        i = maxi ;

        // swap j-th row with i-th row and
        // normalize j-th row
        for ( int jj=j; jj<2; jj++ ){
            std::swap( At(j,jj) , At(i,jj) ) ;
            At(j,jj) /= maxa ;
        }
        std::swap( i_b[j], i_b[i] ) ;
        i_b[j] /= maxa ;

        // elimination
        for ( int ii=j+1; ii<2; ii++ )
        {
            Real_ x = At(ii,j) ;
            for( int jj=j; jj<2; jj++ )
                                        At(ii,jj) -= x*At(j,jj);
            i_b[ii] -= x*i_b[j] ;
        }
    }
    return true;
  }
   Im2D<U_INT2,INT4> circular_window(int radius)
   {
       //define a square
       int side=2*radius+1;
       //Pt2di sz(side,side);


       //=======SET CARRE To zero values
       Pt2di center(radius,radius);
       Im2D<U_INT2,INT4> circle;
       for (int i=0;i<side;i++)
       {
           for (int j=0;j<side;j++)
           {
               Pt2di Point(i,j);
               if((pow(center.x-Point.x,2)+pow(center.y-Point.y,2))<=pow(radius,2))
               {
                   circle.SetI(Point,1);
               }
               else
               {
                   circle.SetI(Point,0);
               }
           }
       }
       return circle;
   }
   
   Im2D<REAL4,REAL8> RefinedCircularWindow(int radius)
   {
       //define a square
        int side=2*radius+1;
        Im2D<REAL4,REAL8> circular=Im2D<REAL4,REAL8>(side,side);
        for(int i=0;i<side;i++)
        {
            for(int j=0;j<side;j++)
            {
                Pt2di Point(i,j);
                circular.set_brd(Point,0);
            }
        }
        //std::cout<<"circular size "<<circular.sz()<<endl;
       // Compute the refined circular window by evaluation integrating 
       // over regions with  partial overlap
		int nbins=36;
		float step = float((0.5* M_PI) / nbins);
        float deltaAngle = 0.0f;
        std::vector<Pt2df> Nodes;
        for (int i=0 ; i<nbins;i++)
		{
            Pt2df Pt;
			Pt.x=radius*cos(deltaAngle);
			Pt.y=radius*sin(deltaAngle);
			Nodes.push_back(Pt);
			deltaAngle+=step;
         }
        int i=0;
		float summ;
        int old_locx=std::floor(Nodes[0].x);
        int old_locy=std::floor(Nodes[0].y);
		int Checkx=Nodes[i].x>=old_locx && Nodes[i].x<old_locx+1;
		int Checky=Nodes[i].y>=old_locy && Nodes[i].y<old_locy+1;
        while(i<(int)Nodes.size())
		{
			summ=0.0;
			i++;
			while(Checkx && Checky)
			{
                old_locx=std::floor(Nodes[i].x);
                old_locy=std::floor(Nodes[i].y);
				summ+=abs(Nodes[i].x-Nodes[i-1].x)*(Nodes[i].y+Nodes[i-1].y-2*old_locy)*0.5;
				/*std::cout<<"***********************\n";
				std::cout<<old_locx<<" "<<old_locy<<endl;
				std::cout<<Nodes[i-1].x<<" "<<Nodes[i-1].y<<endl; 
				std::cout<<Nodes[i].x<<" "<<Nodes[i].y<<endl; 
				std::cout<<"***********************\n";*/
				i++;
				Checkx=Nodes[i].x>=old_locx && Nodes[i].x<old_locx+1;
				Checky=Nodes[i].y>=old_locy && Nodes[i].y<old_locy+1;

			}
			summ+=(Nodes[i-1].x-old_locx)*(Nodes[i-1].y-old_locy);
			//std::cout<<Nodes[i-1].x<<" "<<Nodes[i-1].y<<endl;
			//std::cout<<summ<<endl;
			//std::cout<<"**************************\n";
            Pt2di Loc(old_locx+1+radius,old_locy+1+radius);
            circular.SetR(Loc,(float)summ);
            old_locx=std::floor(Nodes[i].x);
            old_locy=std::floor(Nodes[i].y);
			Checkx=Nodes[i].x>=old_locx && Nodes[i].x<old_locx+1;
			Checky=Nodes[i].y>=old_locy && Nodes[i].y<old_locy+1;
		}

		i=0;
		bool stop=false;
		while(i<=radius)
		{
			int j=0;
			while((j<=radius) && (stop==false))
			{
				float value;
                Pt2di Loc(i+radius,j+radius);
                value=(float)circular.GetR(Loc);
				if(value==0)
                {circular.SetR(Loc,1.0);}
				else
				{stop=true;}
				j++;
			}
			stop=false;
			i++;
		}
		for (int j=0;j<=radius;j++)
			{for (int i=0;i<=radius;i++)
			{
                Pt2di Dst(i+radius,radius-j);
                Pt2di Loc(i+radius,j+radius);
                circular.SetR(Dst,(float)circular.GetR(Loc));
			}
		}
		for (int j=0;j<=radius;j++)
			{for (int i=0;i<=radius;i++)
			{
                Pt2di Dst(radius-i,j);
                Pt2di Loc(i+radius,j);
                circular.SetR(Dst,(float)circular.GetR(Loc));
                //circular.at<float>(j,radius-i)=circular.at<float>(j,i+radius);
			}
		}
		for (int j=0;j<=radius;j++)
			{for (int i=0;i<=radius;i++)
			{
                Pt2di Dst(radius-i,j+radius);
                Pt2di Loc(i+radius,j+radius);
                circular.SetR(Dst,(float)circular.GetR(Loc));
                //circular.at<float>(j+radius,radius-i)=circular.at<float>(j+radius,i+radius);
			}
        }
       return circular;
   }


    void MsdDetector::contextualSelfDissimilarity(Im2D<U_INT2,INT> &img, int xmin, int xmax, float* saliency)
	{
		int r_s = m_patch_radius;
        int r_b = m_search_area_radius;
        int k = m_kNN;

        int w = img.sz().x;
        int h = img.sz().y;
        std::cout<<"Layer width "<<w<<" Layer height "<<h<<endl;


        //ElTimer chrono;
        // Define the circular patch mask that is to be used
        /***************************************************/
        Im2D<REAL4,REAL8> Cir_Mask=RefinedCircularWindow(r_s);
        int side_b = 2 * r_b + 1;
        int border = r_s + r_b;
        int den;

        //bool yescircular=MsdDetector::getCircularWindow();
        int ctrInd=0;
        std::vector<float> minVals(k);
        float *acc = new float[side_b * side_b];

        den = k;// instead of side_s * side_s *PI_4*k


        if (m_circular_window)
        {

        float ab,a_2,b_2;
        for(int y = border; y< h - border; y++)
        {
            for (int x = xmin; x<xmax; x++)
            {
            ctrInd = 0;
            for (int kk = 0; kk<k; kk++)
                minVals[kk] = std::numeric_limits<float>::max();


			//compute central patch values one time for all
            b_2=0;
			/**********************************************************/
           // ElTimer Chroncentpatch;
			for (int u = -r_s; u <= r_s; u++)
			{
				for (int v = -r_s; v <= r_s; v++)
				{
                    Pt2di Loc(u+r_s,v+r_s);
                    float c=(float)Cir_Mask.GetR(Loc);
                    if(c)
                    {
                        //b+=c*img.at<unsigned char>(y + v, x + u);
                        Pt2di LocImg(x+u,y+v);
                        int valxy=img.GetI(LocImg);
                        b_2+=c*valxy*valxy;
                    }
                }
			}
            //std::cout<<" central patch correl "<<Chroncentpatch.uval()<<endl;
			/**********************************************************/
			// Compute the overlapping region patch sum of elements
            for (int j = y - r_b; j <= y + r_b; j++)
            {
                for (int i = x - r_b; i <= x + r_b; i++)
                {
                    if (j == y && i == x)
                        continue;

                    acc[ctrInd] = 0;
                    ab=0;a_2=0;
                    //ElTimer aChrono;
                    //*********************************
                    for (int u = -r_s; u <= r_s; u++)
                    {

                        for (int v = -r_s; v <= r_s; v++)
                        {
                            Pt2di Loc(u+r_s,v+r_s);
                            float c=(float)Cir_Mask.GetR(Loc);
                            Pt2di Pijuv(i+u,j+v);
                            Pt2di Pxyuv(x+u,y+v);
                            if (c)
                            {
                            int valiu = img.GetI(Pijuv);
                            int valxy=img.GetI(Pxyuv);
                            ab+=c*valiu*valxy;
                            a_2+=c*valiu*valiu;
                            }

                        }

                    }
                    acc[ctrInd]=2*(1.0-ab/sqrt(a_2*b_2));              
                   
                   if (acc[ctrInd] < minVals[k - 1])
                    {
                        minVals[k - 1] = acc[ctrInd];
                        for (int kk = k - 2; kk >= 0; kk--)
                        {
                            if (minVals[kk] > minVals[kk + 1])
                            {
                                std::swap(minVals[kk], minVals[kk + 1]);
                            }
                            else
                                break;
                        }
                    }

                    ctrInd++;

                }
            }
            saliency[y*w + x] =  computeAvgDistance(minVals, den);
          }
        }
        }

        else
        {
        float ab,a_2,b_2;
        for(int y = border; y< h - border; y++)
        {
            for (int x = xmin; x<xmax; x++)
            {
            ctrInd = 0;
            for (int kk = 0; kk<k; kk++)
                minVals[kk] = std::numeric_limits<float>::max();

            for (int j = y - r_b; j <= y + r_b; j++)
            {
                for (int i = x - r_b; i <= x + r_b; i++)
                {
                    if (j == y && i == x)
                        continue;

                    acc[ctrInd] = 0;

                    ab=0;a_2=0;b_2=0;

                    //*********************************
                    for (int u = -r_s; u <= r_s; u++)
                    {

                        for (int v = -r_s; v <= r_s; v++)
                        {
                            Pt2di Pijuv(i+u,j+v);
                            Pt2di Pxyuv(x+u,y+v);
                            U_INT2 pijuv=img.GetI(Pijuv);
                            U_INT2 pxyuv=img.GetI(Pxyuv);
                            ab+=pijuv*pxyuv;
                            a_2+=pow(pijuv,2);
                            b_2+=pow(pxyuv,2);
                        }
                    }

                    acc[ctrInd]=2*(1-(ab/pow(a_2*b_2,0.5)));
                    if (acc[ctrInd] < minVals[k - 1])
                    {
                        minVals[k - 1] = acc[ctrInd];
                        for (int kk = k - 2; kk >= 0; kk--)
                        {
                            if (minVals[kk] > minVals[kk + 1])
                            {
                                std::swap(minVals[kk], minVals[kk + 1]);
                            }
                            else
                                break;
                        }
                    }

                    ctrInd++;
                }
            }
            saliency[y*w + x] = computeAvgDistance(minVals, den);
            }
        }
        }

		delete[] acc;
		acc=0;
    }



    float MsdDetector::computeOrientation(Im2D<U_INT2,INT> &img, int x, int y, std::vector<Pt2df> circle)
	{
		int temp;
		int nBins = 36;
		float step = float((2 * M_PI) / nBins);
		std::vector<float> hist(nBins, 0);
		std::vector<int> dists(circle.size(), 0);

		int minDist = std::numeric_limits<int>::max();
		int maxDist = -1;

		for (int k = 0; k<(int)circle.size(); k++)
		{

			int j = y + static_cast <int> (circle[k].y);
			int i = x + static_cast <int> (circle[k].x);

			for (int v = -m_patch_radius; v <= m_patch_radius; v++)
			{
				for (int u = -m_patch_radius; u <= m_patch_radius; u++)
				{
                    Pt2di Pijuv(i+u,j+v);
                    Pt2di Pxyuv(x+u,y+v);
                    temp = img.GetI(Pijuv) - img.GetI(Pxyuv);
					dists[k] += temp*temp;
				}
			}

			if (dists[k] > maxDist)
				maxDist = dists[k];
			if (dists[k] < minDist)
				minDist = dists[k];
		}

		float deltaAngle = 0.0f;
		for (int k = 0; k<(int)circle.size(); k++)
		{
			float angle = deltaAngle;
			float weight = (1.0f*maxDist - dists[k]) / (maxDist - minDist);

			float binF;
			if (angle >= 2 * M_PI)
				binF = 0.0f;
			else
				binF = angle / step;
			int bin = static_cast <int> (std::floor(binF));

			assert(bin >= 0 && bin < nBins);
			float binDist = abs(binF - bin - 0.5f);

			float weightA = weight * (1.0f - binDist);
			float weightB = weight * binDist;
			hist[bin] += weightA;

			if (2 * (binF - bin) < step)
				hist[(bin + nBins - 1) % nBins] += weightB;
			else
				hist[(bin + 1) % nBins] += weightB;

			deltaAngle += step;
		}

		int bestBin = -1;
		float maxBin = -1;
		for (int i = 0; i<nBins; i++)
		{
			if (hist[i] > maxBin)
			{
				maxBin = hist[i];
				bestBin = i;
			}
		}

		//parabolic interpolation
		int l = (bestBin == 0) ? nBins - 1 : bestBin - 1;
		int r = (bestBin + 1) % nBins;
		float bestAngle2 = bestBin + 0.5f * ((hist[l]) - (hist[r])) / ((hist[l]) - 2.0f*(hist[bestBin]) + (hist[r]));
		bestAngle2 = (bestAngle2 < 0) ? nBins + bestAngle2 : (bestAngle2 >= nBins) ? bestAngle2 - nBins : bestAngle2;
		bestAngle2 *= step;
		
		return bestAngle2;
	}

 void MsdDetector::RefineKP( const Pt2di &i_p, Pt2dr &o_p , float * SaliencyMap, int lvl)
     {

        /*
         * This method is intended to refine a keypoint loclation using Taylor series expansion in the space x,y ,scale
         * However, by looking into sift implementations, this is done in each octave separately and therefore scale defines
         * the dgree of blur not the varying size across octaves
         * We use only x,y coordinates to enhance the point position
        */

        /* If scale is to be integrated in our context, it will be defined across levels of pyramid and therefore,
         * therefore derivatives across scales will be computed by regarding saliency at other scales in the vicinty of a point
         * representative scale
         */
         REAL8 m[6];
         REAL8 b[2];
         int x = i_p.x,
             y = i_p.y;
         int Dx=0, Dy=0;
         int iter;
         int m_width=m_scaleSpace[lvl].sz().x;
         int m_height=m_scaleSpace[lvl].sz().y;
         int offset = x+y*m_width;
         SaliencyMap+=offset;
         int c0,c1,c2,c6,c8;
         c6 = m_width-1;
         c8 = m_width+1;
         c1 = -m_width;
         c0 = c1-1;
         c2 = c1+1;

        // U_INT offset;
         //float *itScale0, *itScale1, *itScale2;
         REAL8 dx, dy, dxx, dyy, dxy;
         bool inverted;

         // reiterate until variation is low
         for( iter=0; iter<5; iter++ )
         {
             x += Dx;
             y += Dy;

             dx = 0.5*( SaliencyMap[1]-SaliencyMap[-1] );
             dy = 0.5*( SaliencyMap[m_width]-SaliencyMap[c1] );
             m[2]= b[0] = -dx;
             m[5]= b[1] = -dy;

             m[0]  = dxx = SaliencyMap[1]-SaliencyMap[-1]-( 2.*SaliencyMap[0]);// dxx
             m[4]  = dyy = SaliencyMap[m_width]-SaliencyMap[c1] -( 2.*SaliencyMap[0] ); // dyy

             m[1] = m[3] = dxy = 0.25*( SaliencyMap[c8]+SaliencyMap[c0]-SaliencyMap[c6]-SaliencyMap[c2] ); // dxy

             inverted=Gauss22_invert_b( m, b ); // Normally it is not needed for 2x2 matrix but i use it anyway::" The invert of A is directly obtained"


             if (inverted)

             {
                 // shall we reiterate ?
                 Dx=   ( ( ( b[0]>0.6 ) && ( x<m_width-2 ) )?1:0 )
                     + ( ( ( b[0]<-0.6 ) && ( x>1 ) )?-1:0 );
                 Dy=   ( ( ( b[1]>0.6 ) && ( y<m_height-2 ) )?1:0 )
                     + ( ( ( b[1]<-0.6 ) && ( y>1 ) )?-1:0 );
                 if( Dx == 0 && Dy == 0 ) break;
             }
             else
             {
                 break;
             }
         }
         //Check if Point is not away from origin
         REAL8 xn,yn;
         //check for divergence:: keep original point if there is an issue with refinement
         //Distance between old and new point should not be bigger than 1 pixel

         if (inverted)
         {
             xn = x + b[0] ;
             yn = y + b[1] ;
             // Check if isnan
             if (isnan(xn)||isnan(yn))
             {
                xn=i_p.x;
                yn=i_p.y;
             }
             if (((xn-i_p.x)*(xn-i_p.x)+(yn-i_p.y)*(yn-i_p.y))>1)
             {
                 xn=i_p.x;
                 yn=i_p.y;
             }
             o_p.x= xn;
             o_p.y = yn;
         }
         else
         {
             o_p.x= i_p.x;
             o_p.y = i_p.y;
         }
     }

    void MsdDetector::nonMaximaSuppression(std::vector<float *> & saliency, std::vector<KeyPoint> & keypoints)
	{
        KeyPoint kp_temp;
        //int side = m_search_area_radius * 2 + 1;
        int border = m_search_area_radius + m_patch_radius;
        std::vector<Pt2df> orientPoints;
		if (m_compute_orientation)
		{
			int nBins = 36;
			float step = float((2 * M_PI) / nBins);
			float deltaAngle = 0.0f;

			for (int i = 0; i<nBins; i++)
			{
                Pt2df pt;
				pt.x = m_search_area_radius * cos(deltaAngle);
				pt.y = m_search_area_radius * sin(deltaAngle);

				orientPoints.push_back(pt);

				deltaAngle += step;
			}
		}

		for (int r = 0; r<m_cur_n_scales; r++)
		{
            int cW = m_scaleSpace[r].sz().x;
            int cH = m_scaleSpace[r].sz().y;
		
			for (int j = border; j< cH - border; j++)
			{
				for (int i = border; i< cW - border; i++)
				{

                    if (saliency[r][j * cW + i] > m_th_saliency)
					{
						bool is_max = true;

                        for (int k = fmax(0, r - m_nms_scale_radius); k <= fmin(m_cur_n_scales - 1, r + m_nms_scale_radius); k++)
						{
							if (k != r)
							{
                                int j_sc = (INT)round(j * std::pow(m_scale_factor, r - k));
                                int i_sc = (INT)round(i * std::pow(m_scale_factor, r - k));


                                if (saliency[r][j*cW + i] < saliency[k][j_sc*cW + i_sc])
								{
									is_max = false;
									break;
								}
							}
						}

                        for (int v = fmax(border, j - m_nms_radius); v <= fmin(cH - border - 1, j + m_nms_radius); v++)
						{
                            for (int u = fmax(border, i - m_nms_radius); u <= fmin(cW - border - 1, i + m_nms_radius); u++)
							{

                                if (saliency[r][j*cW + i] < saliency[r][v*cW + u])
								{
                                    is_max = false;
									break;
								}
							}

							if (!is_max)
								break;
						}

                        if (is_max)
                        {
                            if (m_RefinedKps)
                            {
                                Pt2dr Pout;
                                this->RefineKP(Pt2di(i,j),Pout, saliency[r],r);
                                kp_temp.setPointx( Pout.x * std::pow(m_scale_factor, r));
                                kp_temp.setPointy( Pout.y * std::pow(m_scale_factor, r));
                                kp_temp.setResponse(saliency[r][j*cW + i]);
                                kp_temp.setSize((m_patch_radius*2.0f + 1) * std::pow(m_scale_factor, r));

                                if (m_compute_orientation)
                                {
                                    kp_temp.setAngle(computeOrientation(m_scaleSpace[r], i, j, orientPoints));
                                }
                                keypoints.push_back(kp_temp);
                            }
                            else
                            {
                                kp_temp.setPointx( i * std::pow(m_scale_factor, r));
                                kp_temp.setPointy( j * std::pow(m_scale_factor, r));
                                kp_temp.setResponse(saliency[r][j*cW + i]);
                                kp_temp.setSize((m_patch_radius*2.0f + 1) * std::pow(m_scale_factor, r));

                                if (m_compute_orientation)
                                     {
                                       kp_temp.setAngle(computeOrientation(m_scaleSpace[r], i, j, orientPoints));
                                     }

                                keypoints.push_back(kp_temp);
                            }
                        }
					}
				}
			}
		}

	}

    std::vector<KeyPoint> MsdDetector::detect(Tiff_Im &img)
	{
		int border = m_search_area_radius + m_patch_radius;

        //computation of the number of scales
		if (m_n_scales == -1)
            m_cur_n_scales = std::floor(log(fmin(img.sz().x, img.sz().y) / ((m_patch_radius + m_search_area_radius)*2.0 + 1)) / log(m_scale_factor));
		else
			m_cur_n_scales = m_n_scales;

        //std::cout<<"m_cur_n_scales "<<m_cur_n_scales<<endl;
        Im2D<U_INT2,INT> imgG=Im2D<U_INT2,INT>(img.sz().x,img.sz().y);

        if (img.nb_chan() == 1)
          {
              ELISE_COPY(imgG.all_pts(),img.in(),imgG.out());
          }
       else
       //Convert image to GRAY IMAGE
          {
              ELISE_COPY(imgG.all_pts(),(0.21*img.in().v0()+0.72*img.in().v1()+0.07*img.in().v2()),imgG.out());
          }

        ImagePyramid<U_INT2,INT> scaleSpacer=ImagePyramid<U_INT2,INT>(imgG, m_cur_n_scales, m_scale_factor);

        m_scaleSpace = scaleSpacer.getImPyr();

        /***************************************************************/
        std::vector<KeyPoint> keypoints;
        std::vector<float *> saliency;
		saliency.resize(m_cur_n_scales);

		for (int r = 0; r < m_cur_n_scales; r++)
		{
            //std::cout<<m_scaleSpace[r].sz()<<endl;
            saliency[r] = new float[m_scaleSpace[r].sz().y * m_scaleSpace[r].sz().x];
		}

        std::cout<<"==============> Computing saliency maps for the image pyramid\n";
        for (int r = 0; r<m_cur_n_scales; r++)
		{
#ifdef BOOST_MULTICORE
			unsigned nThreads = boost::thread::hardware_concurrency();
			unsigned stepThread = (m_scaleSpace[r].cols - 2 * border) / nThreads;

			std::vector<boost::thread*> threads;
			for (unsigned i = 0; i < nThreads - 1; i++)
			{
				threads.push_back(new boost::thread(&MsdDetector::contextualSelfDissimilarity, this, m_scaleSpace[r], border + i*stepThread, border + (i + 1)*stepThread, saliency[r]));
			}
            threads.push_back(new boost::thread(&MsdDetector::contextualSelfDissimilarity, this, m_scaleSpace[r], border + (nThreads - 1)*stepThread, m_scaleSpace[r].sz().x - border, saliency[r]));

			for (unsigned i = 0; i < threads.size(); i++)
			{
				threads[i]->join();
				delete threads[i];
			}
#else
            ElTimer chrono;
            contextualSelfDissimilarity(m_scaleSpace.at(r), border, m_scaleSpace.at(r).sz().x - border, saliency[r]);
            //m_scaleSpace.erase(m_scaleSpace.begin());
            std::cout<<" Time elapsed for layer computation: "<<chrono.uval()<<endl;

#endif
		}

		nonMaximaSuppression(saliency, keypoints);

		for (int r = 0; r<m_n_scales; r++)
		{
			delete[] saliency[r];
		}

		m_scaleSpace.clear();

		return keypoints;
	}
