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

#include "StdAfx.h"
#include "msdImgPyramid.h"
#include "Image.h"



template <class Type, class TyBase>
ImagePyramid<Type,TyBase>::ImagePyramid( Im2D<Type,TyBase>  & im, const int nLevels, const float scaleFactor)
{
	m_nLevels = nLevels;
	m_scaleFactor = scaleFactor;
	m_imPyr.clear();
	m_imPyr.resize(nLevels);
	
    m_imPyr[0]=Im2D<Type,TyBase>(im.sz().x,im.sz().y);
	
	ELISE_COPY(m_imPyr[0].all_pts(),im.in(),m_imPyr[0].out());

    // Palette Initialization
   // Elise_Set_Of_Palette SOP=Elise_Set_Of_Palette::TheFullPalette();/*(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));*/


   // std::cout<<Elise_Palette(Pgr).<<endl;

    // Creation of video windows
   /* Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);
    Video_Win W (Ecr,SOP,Pt2di(50,50),Pt2di(m_imPyr[0].sz().x, m_imPyr[0].sz().y));

    Im2D_U_INT2 I(m_imPyr[0].sz().x, m_imPyr[0].sz().y);
    // chargement du fichier dans l’image  et la fenetre

    ELISE_COPY
    (

    I.all_pts(),
    m_imPyr[0].in(),
    I.out()|W.ogray()

    );*/


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


