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
#define MSD_ORIENTATION_NB_BINS 36

#include "msd.h"


Fonc_Num sobel_1(Fonc_Num f)
{
    Im2D_REAL8 Fx
               (  3,3,
                  " -1 0 1 "
                  " -2 0 2 "
                  " -1 0 1 "
                );
    Im2D_REAL8 Fy
               (  3,3,
                  " -1 -2 -1 "
                  "  0  0  0 "
                  "  1  2  1 "
                );
   return
       Abs(som_masq(f,Fx,Pt2di(-1,-1)))
     + Abs(som_masq(f,Fy));
}


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


void MsdDetector::contextualSelfDissimilarity(Im2D<U_INT1,INT> &img, int xmin, int xmax, float* saliency)
{
    int r_s = m_patch_radius;
    int r_b = m_search_area_radius;
    int k = m_kNN;
    int w = img.sz().x;
    int h = img.sz().y;

    int side_b = 2 * r_b + 1;
    // apparently, a patch of 3x3 compared in a search region of 5x5 will compare the 3x3 centered on -5,0 eg , thus the border is indeed Search region + patch radius
    int border = r_s + r_b;
    int a, b,B_2;
    int den = k;

    std::vector<float> minVals(k);
    int *accAB = new int[side_b * side_b];
    int *accA_2 = new int[side_b * side_b];
    float acc=0.0;
    int **vAB = new int *[w];
    int **vA_2= new int *[w];
    for (int i = 0; i<w; i++)
    {
        vAB[i]  = new int[side_b * side_b];
        vA_2[i] = new int[side_b * side_b];
    }
    int ctrInd;

    Im2D<U_INT2,INT> SquaredIm(img.sz().x,img.sz().y,0);
    ELISE_COPY(img.all_pts(),Square(img.in()),SquaredIm.out());
    // sum of square in a rectangle
    Im2D<REAL4,REAL8> SOSIm(img.sz().x,img.sz().y,0.0);
    ELISE_COPY(img.all_pts(),rect_som(SquaredIm.in_proj(),Pt2di(2*r_s,2*r_s)),SOSIm.out());

    if (mDebug) std::cout << "initialize saliency on first image position";
    //first position
    int x = xmin;
    int y = border;
    B_2=0;

    // sum of square in the patch
    ELISE_COPY(rectangle(Pt2di(x-r_s,y-r_s),Pt2di(x+r_s+1,y+r_s+1)), SquaredIm.in(),sigma(B_2));

    ctrInd = 0;
    for (int kk = 0; kk<k; kk++)
        minVals[kk] = std::numeric_limits<float>::max();

    for (int j = y - r_b; j <= y + r_b; j++)
    {
        for (int i = x - r_b; i <= x + r_b; i++)
        {
            if (j == y && i == x)
                continue;

            accAB[ctrInd] = 0;
            accA_2[ctrInd] =0;
            for (int u = -r_s; u <= r_s; u++)
            {
                vAB[x + u][ctrInd] = 0;
                vA_2[x + u][ctrInd] = 0;
                for (int v = -r_s; v <= r_s; v++)
                {
                    Pt2di Pijuv(i+u,j+v);
                    Pt2di Pxyuv(x+u,y+v);
                    a = img.GetI(Pijuv);
                    b = img.GetI(Pxyuv);
                    vAB[x + u][ctrInd]  += (a*b);
                    vA_2[x + u][ctrInd] += (a*a);
                }
                accAB[ctrInd] += vAB[x + u][ctrInd];
                //std::cout<<" acc ab "<<accAB[ctrInd]<<endl;
                accA_2[ctrInd] += vA_2[x + u][ctrInd];
                //std::cout<<"acc a_2 "<<accA_2[ctrInd]<<endl;
            }
            //Get the new distance based on NCC
            //std:: cout<< accAB[ctrInd] <<"  "<< accA_2[ctrInd]<<"  "<<B_2<<endl;

            acc=2*(1.0-(float)accAB[ctrInd]/sqrt((float)accA_2[ctrInd]*(float)B_2));
            //std::cout<<" acc  : "<<acc<<endl;

            if (acc  < minVals[k - 1])
            {
                minVals[k - 1] = acc;

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
    if (mDebug) std::cout << " ...done\n";


    if (mDebug) std::cout << "Compute saliency for first row only ";
    for (x = xmin + 1; x<xmax; x++)
    {
        //Compute the central patch sum of square values

        B_2=0;
        ELISE_COPY(rectangle(Pt2di(x-r_s,y-r_s),Pt2di(x+r_s+1,y+r_s+1)),SquaredIm.in(),sigma(B_2));

        // We could use the previously computed values

        ctrInd = 0;
        for (int kk = 0; kk<k; kk++)
            minVals[kk] = std::numeric_limits<float>::max();

        for (int j = y - r_b; j <= y + r_b; j++)
        {
            for (int i = x - r_b; i <= x + r_b; i++)
            {
                if (j == y && i == x)
                    continue;

                vAB[x + r_s][ctrInd] = 0;
                vA_2[x + r_s][ctrInd] = 0;

                for (int v = -r_s; v <= r_s; v++)
                {
                    Pt2di Pijuv(i+r_s,j+v);
                    Pt2di Pxyuv(x+r_s,y+v);
                    a = img.GetI(Pijuv);
                    b = img.GetI(Pxyuv);
                    vAB[x + r_s][ctrInd]  += (a*b);
                    vA_2[x + r_s][ctrInd] += (a*a);
                }

                accAB[ctrInd] = accAB[ctrInd] + vAB[x + r_s][ctrInd] - vAB[x - r_s - 1][ctrInd];
                accA_2[ctrInd] = accA_2[ctrInd] + vA_2[x + r_s][ctrInd] - vA_2[x - r_s - 1][ctrInd];

                //Get the new distance based on NCC

                acc=2*(1.0-(float)accAB[ctrInd]/sqrt((float)accA_2[ctrInd]*(float)B_2));

                //std::cout<<" acc  : "<<acc<<endl;
                if (acc < minVals[k - 1])
                {
                    minVals[k - 1] = acc;
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

    if (mDebug) std::cout << " ...done\n";
    if (mDebug) std::cout << "start saliency computation on all rows \n";

    for (int y = border + 1; y< h - border; y++)
    {
        ctrInd = 0;
        for (int kk = 0; kk<k; kk++)
            minVals[kk] = std::numeric_limits<float>::max();
        x = xmin;

        //Compute the central patch sum of square values
        B_2=0;
        ELISE_COPY(rectangle(Pt2di(x-r_s,y-r_s),Pt2di(x+r_s+1,y+r_s+1)), SquaredIm.in(),sigma(B_2));

        for (int j = y - r_b; j <= y + r_b; j++)
        {
            for (int i = x - r_b; i <= x + r_b; i++)
            {
                if (j == y && i == x)
                    continue;

                accAB[ctrInd] = 0;
                accA_2[ctrInd]= 0;

                for (int u = -r_s; u <= r_s; u++)
                {
                    Pt2di Pijuv(i+u,j+r_s);
                    Pt2di Pxyuv(x+u,y+r_s);

                    a = img.GetI(Pijuv);
                    b = img.GetI(Pxyuv);
                    vAB[x + u][ctrInd]  += (a*b);
                    vA_2[x + u][ctrInd] += (a*a);
                    Pijuv.x=i+u; Pijuv.y=j-r_s-1;
                    Pxyuv.x=x+u; Pxyuv.y=y-r_s-1;
                    a = img.GetI(Pijuv);
                    b = img.GetI(Pxyuv);
                    vAB[x + u][ctrInd]  -= (a*b);
                    vA_2[x + u][ctrInd] -= (a*a);

                    accAB[ctrInd] += vAB[x + u][ctrInd];
                    accA_2[ctrInd] += vA_2[x + u][ctrInd];
                }

                acc=2*(1.0-(float)accAB[ctrInd]/sqrt((float)accA_2[ctrInd]*(float)B_2));
                //std::cout<<" acc  : "<<acc<<endl;
                if (acc  < minVals[k - 1])
                {
                    minVals[k - 1] = acc;

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

        //loop on each x
        for (x = xmin + 1; x<xmax; x++)
        {
            //Compute the central patch sum of square values
            B_2=0;
            ELISE_COPY(rectangle(Pt2di(x-r_s,y-r_s),Pt2di(x+r_s+1,y+r_s+1)), SquaredIm.in(),sigma(B_2));

            ctrInd = 0;
            for (int kk = 0; kk<k; kk++)
                minVals[kk] = std::numeric_limits<float>::max();

            for (int j = y - r_b; j <= y + r_b; j++)
            {
                for (int i = x - r_b; i <= x + r_b; i++)
                {
                    if (j == y && i == x)
                        continue;

                    Pt2di Pijuv(i+r_s,j+r_s);
                    Pt2di Pxyuv(x+r_s,y+r_s);
                    a = img.GetI(Pijuv);
                    b = img.GetI(Pxyuv);
                    vAB[x + r_s][ctrInd]  += (a*b);
                    vA_2[x + r_s][ctrInd] += (a*a);

                    Pijuv.x=i+r_s; Pijuv.y=j-r_s-1;
                    Pxyuv.x=x+r_s; Pxyuv.y=y-r_s-1;

                    a = img.GetI(Pijuv);
                    b = img.GetI(Pxyuv);
                    vAB[x + r_s][ctrInd]  -= (a*b);
                    vA_2[x + r_s][ctrInd] -= (a*a);

                    accAB[ctrInd] = accAB[ctrInd] + vAB[x + r_s][ctrInd] - vAB[x - r_s - 1][ctrInd];

                    accA_2[ctrInd] = accA_2[ctrInd] + vA_2[x + r_s][ctrInd] - vA_2[x - r_s - 1][ctrInd];

                    acc=2*(1.0-(float)accAB[ctrInd]/sqrt((float)accA_2[ctrInd]*(float)B_2));
                    //std::cout<<" acc  : "<<acc<<endl;
                    if (acc < minVals[k - 1])
                    {
                        minVals[k - 1] = acc;

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

    // delete
    for (int i = 0; i<w; i++)
    {
        delete[] vAB[i];
        delete[] vA_2[i];
    }
    delete[] vAB;
    delete[] vA_2;
    delete[] accAB;
    delete[] accA_2;

}




void MsdDetector::nonMaximaSuppression(std::vector<float *> & saliency)
{

    //int side = m_search_area_radius * 2 + 1;
    //int border = m_search_area_radius + m_patch_radius;
    // saliency map has already a border of m_search_area_radius + m_patch_radius, so i double it otherwise there are a lot of kp on the edge (because there are local maximum)

    int border = 2*(m_search_area_radius + m_patch_radius);

    for (int r = 0; r<m_cur_n_scales; r++)
    {

        mVVKP.push_back(new std::vector<cPtsCaracMSD>);
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
                        cPtsCaracMSD kp_temp;
                        double x= i * std::pow(m_scale_factor, r);
                        double y= j * std::pow(m_scale_factor, r);
                        kp_temp.mPt=Pt2dr(x,y);
                        kp_temp.mPtSc=Pt2dr(i,j);
                        kp_temp.mSize=(m_patch_radius*2.0f + 1) * std::pow(m_scale_factor, r);
                        kp_temp.mScale=std::pow(m_scale_factor, r);
                        mVVKP.at(r)->push_back(kp_temp);
                    }
                }
            }
        }
        if (mDebug) std::cout << "For scale " << r << ", MSD have found " << mVVKP.at(r)->size() << " caract points\n";
    }
}

// create the pyramid of saliency Im2D for export and visualisation purpose
void MsdDetector::saliency2Im2D(const std::vector<float *> &saliency){

    if (mDebug) std::cout << "Export saliency to ram Image ";
    for (int r = 0; r<m_cur_n_scales; r++)
    {

        Pt2di Sz(m_scaleSpace[r].sz());

        Im2D<REAL4,REAL8> saliencyIm2D(Sz.x,Sz.y,0.0);
        for (int row  (0) ; row <Sz.y;row++){
            for (int col (0) ; col <Sz.x;col++){
                Pt2di pt(col,row);
                double aVal=saliency[r][(Sz.x*(row)+col)];
                if (aVal>0.5) aVal=0.5;
                if (aVal<0.0) aVal=0.0;
                saliencyIm2D.SetR(pt,aVal);
            }
        }
        mSaliencyIm.push_back(saliencyIm2D);
    }
    if (mDebug) std::cout << " done \n";
}


template void MsdDetector::detect<U_INT1,INT>(Im2D<U_INT1,INT> &img);

template <class Type, class TyBase>
ImagePyramid<Type,TyBase>::ImagePyramid( Im2D<Type,TyBase>  & im, const int nLevels, const float scaleFactor)
{
    m_nLevels = nLevels;
    m_scaleFactor = scaleFactor;
    m_imPyr.clear();
    m_imPyr.resize(nLevels);

    m_imPyr[0]=Im2D<Type,TyBase>(im.sz().x,im.sz().y);

    ELISE_COPY(m_imPyr[0].all_pts(),im.in(),m_imPyr[0].out());

    if(m_nLevels > 1)
    {
        for (int lvl = 1; lvl < m_nLevels; lvl++)
        {
            float scale = 1 / std::pow(scaleFactor, (float)lvl);
            Pt2dr Newsize(round(im.sz().x*scale),round(im.sz().y*scale));
            m_imPyr[lvl].Resize(Pt2di(Newsize));
            m_imPyr[lvl]=this->resize(im,Newsize);
        }
    }
}

template <class Type,class TyBase>
Im2D<Type, TyBase> ImagePyramid<Type,TyBase>::resize(Im2D<Type,TyBase> & im, Pt2dr Newsize)
{
    cInterpolPPV<Type> * Interpol= new cInterpolPPV<Type>;

    Im2D<Type,TyBase> Out;
    Out.Resize(Pt2di(Newsize));
    float tx=im.sz().x/Newsize.x;
    float ty=im.sz().y/Newsize.y;

    for (int i=0;i<Newsize.x;i++)
    {
        for(int j=0;j<Newsize.y;j++)
        {
            Pt2dr PP(tx*i,ty*j);
            Pt2di Dst(i,j);
            REAL8 RetVal;
            RetVal=im.Get(PP,*Interpol,0);
            if(RetVal<im.vmin() || RetVal>im.vmax())
            {
                RetVal=im.vmax();
            }
            Out.SetI(Dst,round(RetVal));
        }
    }

    return Out;
}

template <class Type, class TyBase>
ImagePyramid<Type,TyBase>::~ImagePyramid()
{
}

template class ImagePyramid<U_INT1,INT>;
template class ImagePyramid<U_INT2,INT>;

// return the number of possible orientations (at most DIGEO_NB_MAX_ANGLES) and the angles (o_angles), providing a gradient (orientation and magnitude) and a point
int MsdDetector::orientate( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracMSD &i_p, REAL8 o_angles[DIGEO_MAX_NB_ANGLES] )
{
    static REAL8 histo[MSD_ORIENTATION_NB_BINS];

    int xi = ((int) (i_p.mPt.x+0.5)) ;
    int yi = ((int) (i_p.mPt.y+0.5)) ;
    // MSD inner method: compute ori on image of the pyram where pt is detected with Search area Radius as size
    //const REAL8 sigmaw = DIGEO_ORIENTATION_WINDOW_FACTOR*i_p.getScale();
    const REAL8 sigmaw = DIGEO_ORIENTATION_WINDOW_FACTOR*i_p.mSize/2;

    int W = (int)ceil( 3*sigmaw );
    // the "orientation collection region", is equal to the size of the kernel for Gaussian Blur of amount 1.5*sigma.
    //W=(int)ceil(1.5*(m_patch_radius+m_search_area_radius));
    // fill the SIFT histogram
    const INT width  = i_gradient.sz().x/2,
            height = i_gradient.sz().y;
    REAL8 dx, dy, r2,
            wgt, mod, ang;
    int   offset;
    const REAL4 *p = i_gradient.data_lin()+( xi+yi*width )*2;

    std::fill( histo, histo+MSD_ORIENTATION_NB_BINS, 0 );
    for ( int ys=std::max( -W, 1-yi ); ys<=std::min( W, height-2-yi ); ys++ )
    {
        for ( int xs=std::max( -W, 1-xi ); xs<=std::min( W, width-2-xi ); xs++ )
        {
            dx = xi+xs-i_p.mPt.x;
            dy = yi+ys-i_p.mPt.y;
            r2 = dx*dx+dy*dy;

            // limit to a circular window
            if ( r2>=W*W+0.5 ) continue;
            // weigthing proportionnal to distance from center
            wgt    = ::exp( -r2/( 2*sigmaw*sigmaw ) );
            offset = ( xs+ys*width )*2;
            // gradient magnitude
            mod    = p[offset];
            //std::cout << "grad magnitude is " << mod <<"\n";
            //  gradient orientation, in radians
            ang    = p[offset+1];

            int bin = (int)floor( MSD_ORIENTATION_NB_BINS*ang/( 2*M_PI ) );
            histo[bin] += mod*wgt;
        }
    }

    REAL8 prev;
    // smooth histogram
    // mean of a bin and its two neighbour values (x3)
    REAL8 *itHisto,
            first, mean;
    int iHisto,
            iIter = 3;
    while ( iIter-- )
    {
        itHisto = histo;
        iHisto  = MSD_ORIENTATION_NB_BINS-2;
        first = prev = *itHisto;
        *itHisto = ( histo[MSD_ORIENTATION_NB_BINS-1]+( *itHisto )+itHisto[1] )/3.; itHisto++;
        while ( iHisto-- )
        {
            mean = ( prev+(*itHisto)+itHisto[1] )/3.;
            prev = *itHisto;
            *itHisto++ = mean;
        }
        *itHisto = ( prev+( *itHisto )+first )/3.; itHisto++;
    }

    // find histogram's peaks
    // peaks are values > 80% of histoMax and > to both its neighbours
    REAL8 histoMax = 0.8*( *std::max_element( histo, histo+MSD_ORIENTATION_NB_BINS ) ),
            v, next, di;
    int nbAngles = 0;
    for ( int i=0; i<MSD_ORIENTATION_NB_BINS; i++ )
    {
        v = histo[i];
        prev = histo[ ( i==0 )?MSD_ORIENTATION_NB_BINS-1:i-1 ];
        next = histo[ ( i==( MSD_ORIENTATION_NB_BINS-1 ) )?0:i+1 ];
        if ( ( v>histoMax ) && ( v>prev ) && ( v>next ) )
        {
            // we found a peak
            // compute angle by quadratic interpolation
            di = -0.5*( next-prev )/( next+prev-2*v );
            o_angles[nbAngles++] = 2*M_PI*( i+di+0.5 )/MSD_ORIENTATION_NB_BINS;
            if ( nbAngles==DIGEO_MAX_NB_ANGLES ) return DIGEO_MAX_NB_ANGLES;
        }
    }
    return nbAngles;
}

// img is the gradient (magnitude and angle)
void MsdDetector::orientate(Im2D<REAL4,REAL8> &img, std::vector<cPtsCaracMSD> & aVPCar)
{
    REAL8 angles[DIGEO_MAX_NB_ANGLES];
    unsigned int nbAngles;

    for (auto & kp : aVPCar){
        nbAngles = orientate(img, kp, angles);
        // most point seems to have more than one orientation, which is bad sign.
        for (unsigned int i(0); i<nbAngles;i++){
            if ((kp.getAngles().size()-1) >= i) { kp.setAngle(angles[i],i);}
            else { kp.addAngle(angles[i]);}
        }
    }
}

// work on u_int2 but no keypoints so useless. I stop implementing a compatibility with 16 bits images, only 8 bits right now
template <class Type, class TyBase>
void MsdDetector::detect(Im2D<Type,TyBase> &img)
{

    if (mDebug) std::cout<<"MSD detector: start\n";
    int border = m_search_area_radius + m_patch_radius;

    //computation of the number of scales
    if (m_n_scales == -1){
        int min_NbPix(50);
        m_cur_n_scales = std::floor(log(fmin(img.sz().x, img.sz().y) / ((m_patch_radius + m_search_area_radius + m_nms_radius)*2.0 + min_NbPix)) / log(m_scale_factor));
    }else
        m_cur_n_scales = m_n_scales;
    if (mDebug) std::cout<<"m_cur_n_scales "<<m_cur_n_scales<<endl;

    // vector of pyram images
    ImagePyramid<Type,TyBase> scaleSpacer=ImagePyramid<Type,TyBase>(img, m_cur_n_scales, m_scale_factor);
    // not well implemented, do a copy of the pyram
    m_scaleSpace = scaleSpacer.getImPyr();

    std::vector<float *> saliency;
    saliency.resize(m_cur_n_scales);

    // compute saliency for all scale
    for (int r = 0; r < m_cur_n_scales; r++)
    {
        if (mDebug) std::cout<< "MSD: I process the pyram scale " << r << " , image size is " << m_scaleSpace[r].sz()<<endl;
        saliency[r] = new float[m_scaleSpace[r].sz().y * m_scaleSpace[r].sz().x];
        contextualSelfDissimilarity(m_scaleSpace.at(r), border, m_scaleSpace.at(r).sz().x - border, saliency[r]);
        if (mDebug) std::cout<< "done " <<endl;

    }
    // create Im2D pyram from saliency vector
    saliency2Im2D(saliency);
    // fill the vector of Caract point for each scale by selecting local maximum, remove carct point too closed from edge (edge effect)
    if (mDebug){ std::cout << "detect local maximum on saliency map \n";}
    nonMaximaSuppression(saliency);

    if (mDebug){
        for (int r = 0; r<m_cur_n_scales; r++)
        {
            std::string saliencyName(mTmpDir + "/MSD_Saliency_DZ"+ ToString(r) +"_"+mNameIm+".tif");
            ELISE_fp::RmFileIfExist(saliencyName);

            Im2D_REAL4 aImgSalResized(img.sz().x,img.sz().y,0.0);
            Tiff_Im  aTifOut
                    (
                        saliencyName.c_str(),
                        aImgSalResized.sz(),
                        GenIm::real4,
                        Tiff_Im::No_Compr,
                        Tiff_Im::BlackIsZero
                        );
            double aFact=pow(m_scale_factor,r);
            ELISE_COPY(
                        aImgSalResized.all_pts(),
                        StdFoncChScale(mSaliencyIm.at(r).in_proj(),Pt2dr(0,0),Pt2dr(1/aFact,1/aFact)),
                        aImgSalResized.oclip()
                        );

            ELISE_COPY(aImgSalResized.all_pts(),aImgSalResized.in(),aTifOut.out());

            std::cout << "I write Saliency map to " << saliencyName << "\n";
        }
    }

    // now that caract points has been computed,
    // optionnal : filtering
    // compute orientations
    // compute descriptor and convert to digeo pt

    /* test : tag pt that are detected at two scale
    for (int r = 1; r<m_cur_n_scales; r++)
    {
        for (auto & aPt1: *mVVKP.at(r)){
            // loop on previous scales
            for (int r2 = 0; r2<r; r2++)
            {
                 for (auto & aPt2: *mVVKP.at(r2)){
                  double dist=euclid(aPt1.mPt,aPt2.mPt);
                  if (dist<m_nms_radius+0.5){
                      aPt1.mMulSc=1;
                      aPt2.mMulSc=1;
                  }
                }
            }
        }
    }*/

    orientationAndDescriptor();

    // display all kp on the input image with orientation and scale
    if (mDebug) doIllu(img);

    for (int r = 0; r<m_cur_n_scales; r++)
    {
        delete[] saliency[r];
    }

    m_scaleSpace.clear();

}


template <class Type, class TyBase>
void MsdDetector::doIllu(Im2D<Type, TyBase> &img){

    std::string Name(mTmpDir + "/MSD_kp_"+mNameIm+".tif");
    ELISE_fp::RmFileIfExist(Name);
    Im2D<Type, TyBase> img2(img.sz().x,img.sz().y,0);
    ELISE_COPY(img.all_pts(),img.in(),img2.out());

    for (int r = 0; r<m_cur_n_scales; r++){
        for (auto & kp : *mVVKP.at(r)){
            // draw the circle around pt
            ELISE_COPY(ellipse(kp.mPt,kp.mSize/2,kp.mSize/2,1),
                       200.0,
                       img2.out());
            // draw the point
            ELISE_COPY(ellipse(kp.mPt,1,1,1),
                       250.0,
                       img2.out());
            // draw segment for orientation
            for (auto & angle : kp.getAngles()){
                ELISE_COPY(line(Pt2di(kp.mPt),Pt2di(kp.mPt.x+kp.mSize*0.5*cos(angle),kp.mPt.y+kp.mSize*0.5*sin(angle))),
                           250.0,
                           img2.out());
            }
        }
    }

    std::cout << "I write an illu of keypoints on " << Name << "\n";
    Tiff_Im  aTifOut
            (
                Name.c_str(),
                img2.sz(),
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
                );
    ELISE_COPY(img2.all_pts(),img2.in(),aTifOut.out());

}

template void MsdDetector::doIllu<INT1, INT>(Im2D<INT1, INT> &img);


void MsdDetector::orientationAndDescriptor(){

    if (mDebug) { std::cout << "compute orientation and descriptor ";}
    REAL8 maxValGrad=1;
    //orientation on the raw image
    // compute gradient of image
    Im2D<REAL4,REAL8> i_grad;
    gradient(m_scaleSpace.at(0), maxValGrad, i_grad)  ;
    // compute descriptor
    //DescriptorExtractor<REAL4,REAL8> aDesc=DescriptorExtractor<REAL4,REAL8>(mSaliencyIm.at(0));
    DescriptorExtractor<U_INT1,INT> aDesc=DescriptorExtractor<U_INT1,INT>(m_scaleSpace.at(0));


    for (int r = 0; r<m_cur_n_scales; r++)
    {
        if (mDebug){ std::cout << " scale " << r;}


        //compute orientation for Caract Pt
        orientate(i_grad, *mVVKP.at(r));

        for (auto & aPt: *mVVKP.at(r)){

            DigeoPoint DP;
            DP.x=aPt.mPt.x;
            DP.y=aPt.mPt.y;
            REAL8 scale(aPt.mSize/2);// local scale in lowe convention, what is it?

            //REAL8 scale((m_patch_radius+m_search_area_radius)*aPt.mScale);
            //std::cout << "scale : " << scale << ",pt scale " << aPt.mScale << " , Size/2 " << aPt.mSize/2 << "\n";

            for (auto & angle : aPt.getAngles()){
                REAL8  descriptor[DIGEO_DESCRIPTOR_SIZE];
                aDesc.describe(DP.x,DP.y,scale,angle,descriptor);
                aDesc.normalize_and_truncate(descriptor);
                DP.addDescriptor(angle,descriptor);
            }
            // def constructor of cPtsCaract set angle to 0
            if (DP.angle(0)!=0.0 && DP.nbAngles()<4) mVDP.push_back(DP);
        }
    }

    if (mDebug){ std::cout << " done \n " ;}
}



// method from original MSD code, not used in micmac, adapted or removed


/*
// input: the image, x y= pos of the keypoint, circle; a vector containing 36 Pt2df (for the 36 bin) which are positionned on a circle centered on 0,0
float MsdDetector::computeOrientation(Im2D<U_INT1,INT> &img, int x, int y, std::vector<Pt2df> circle)
{
    int temp;
    int nBins = 36;
    float step = float((2 * M_PI) / nBins);
    std::vector<float> hist(nBins, 0);
    std::vector<int> dists(circle.size(), 0);

    int minDist = std::numeric_limits<int>::max();
    int maxDist = -1;

    // determine min and max distance in radiometry along all diameter of the circle and populate dists[k]
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


          This method is intended to refine a keypoint loclation using Taylor series expansion in the space x,y ,scale
          However, by looking into sift implementations, this is done in each octave separately and therefore scale defines
          the dgree of blur not the varying size across octaves
          We use only x,y coordinates to enhance the point position


     If scale is to be integrated in our context, it will be defined across levels of pyramid and therefore,
         * therefore derivatives across scales will be computed by regarding saliency at other scales in the vicinty of a point
         * representative scale

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
        if (std::isnan(xn)||std::isnan(yn))
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
*/
/*
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
            std::cout<<"***********************\n";
                std::cout<<old_locx<<" "<<old_locy<<endl;
                std::cout<<Nodes[i-1].x<<" "<<Nodes[i-1].y<<endl;
                std::cout<<Nodes[i].x<<" "<<Nodes[i].y<<endl;
                std::cout<<"***********************\n";
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
*/
