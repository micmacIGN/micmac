/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#ifndef __FILTRE_H__
#define __FILTRE_H__

#include <vector>
#include <complex>

template
<class T>
void Gauss(T * DataLin,
        std::complex<int> SzIm,
        std::complex<int> P0,
        std::complex<int> Sz)
{
    std::cout << "Gauss" << std::endl;
    std::vector<float> tempdata;
    tempdata.resize(Sz.real()*Sz.imag());
    float * itout = &tempdata[0];
    for(int l=0;l<Sz.imag();++l)
    {
        T * itin  = &DataLin[(P0.imag()+l)*SzIm.real()+P0.real()];
        T p3;
        T p2 = (*itin);++itin;
        T p1 = 0;
        for(int c=1;c<Sz.real();++c,++itin,++itout)
        {
            p3 = (*itin);
            *(itout) = p1 + 3*p2 + p3;
            p1 = p2;
            p2 = p3;
        }
        *(itout) = p1 + 3*p2;++itout;
    }
    for(int c=0;c<Sz.real();++c)
    {
        float p3;
        float p2 = tempdata[c];
        float p1 = 0.;
        int l;
        for(l=1;l<Sz.imag();++l)
        {
            p3 = tempdata[l*Sz.real()+c];
            DataLin[(P0.imag()+l-1)*SzIm.real()+P0.real()+c]= (T)((p1 + 3*p2 + p3)/25);
            p1 = p2;
            p2 = p3;
        }
        DataLin[(P0.imag()+l-1)*SzIm.real()+P0.real()+c]= (T)((p1 + 3*p2)/25);
    }
}

template
<class T>
void Gauss(T * DataLin,int nBands,
        std::complex<int> SzIm,
        std::complex<int> P0,
        std::complex<int> Sz)
{
    std::cout << "Gauss" << std::endl;
    float * tempdata = new float[Sz.real()*Sz.imag()*nBands];
    float * itout = tempdata;
    for(int l=0;l<Sz.imag();++l)
    {
        T * itin  = &DataLin[(P0.imag()+l)*SzIm.real()*nBands+P0.real()*nBands];
        T * pt2 = itin;
        T * pt3 = pt2 + nBands;
	T * pt1 = pt2;
	for(int c=0;c<nBands;++c,++itout,++pt2,++pt3)
	{
		*itout = 3*(*pt2)+(*pt3);
	}
	for(int c=nBands;c<(Sz.real()-1)*nBands;++c,++itout,++pt1,++pt2,++pt3)
	{
		*itout = (*pt1) + 3*(*pt2) + (*pt3);
	}
	for(int c=0;c<nBands;++c,++itout,++pt1,++pt2)
	{
		*itout = (*pt1) + 3*(*pt2);
	}
    }
    for(int c=0;c<Sz.real()*nBands;++c)
    {
        float p3;
        float p2 = tempdata[c];
        float p1 = 0.;
        int l;
	int Coefc = 5;
        if ((c<nBands)||(c>=(Sz.real()-1)*nBands))
		Coefc = 4;
        for(l=1;l<Sz.imag();++l)
        {
            p3 = tempdata[l*Sz.real()*nBands+c];
	    int Coefl = 5;
	    if (l==1)
		Coefl = 4;
            DataLin[(P0.imag()+l-1)*SzIm.real()*nBands+P0.real()*nBands+c]= (T)((p1 + 3*p2 + p3)/(Coefc*Coefl));
            p1 = p2;
            p2 = p3;
        }
        DataLin[(P0.imag()+l-1)*SzIm.real()*nBands+P0.real()*nBands+c]= (T)((p1 + 3*p2)/(Coefc*4));
    }
   delete tempdata;
}


#endif

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
