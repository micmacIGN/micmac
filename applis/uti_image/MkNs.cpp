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
#include "general/all.h"
#include "private/all.h"

#include <fstream>



typedef enum
{
     Header,
     ElCpp,
     AppliCpp
} MODE;


INT last_line_include(const char * name)
{
    INT res = -1;

    std::ifstream  Cin( name);
    std::string buf;

    INT nbl=0;
    while(getline(Cin,buf))
    {
         if (buf.find("#include")!= std::string::npos)
            res = nbl;
         nbl++;
    }

    return res;
}

class FileCode : public ElActionParseDir
{

    public :
       
       void act(const ElResParseDir & res) ;
       
       FileCode(MODE aMode) : mMode(aMode) {}


    private :
       MODE mMode;
};


void FileCode::act(const ElResParseDir & res)
{
     if (res.is_dir())
        return;


    {
       INT LL_incl = last_line_include(res.name());


       ifstream  Cin( res.name());

       ofstream  Cout( "Tampon");
     
       string buf;
       INT nbl=0;
       if (mMode == Header)
          Cout << "NS_BEGIN_eLiSe  //*-*-*-*-*-*-* = namespace eLiSe {\n";
       while(getline(Cin,buf))
       {
           Cout << buf << "\n";
           if (nbl== LL_incl)
           {
               if (mMode == ElCpp)
                  Cout << "NS_BEGIN_eLiSe  //*-*-*-*-*-*-* =  namespace eLiSe {\n";
               if (mMode == AppliCpp)
                  Cout << "NS_USING_eLiSe //*-*-*-*-*-*-* =  using namespace eLiSe;\n";
           }
           nbl++;
       }
       if ((mMode == Header) || (mMode == ElCpp))
          Cout << "NS_END_eLiSe  //*-*-*-*-*-*-* =  }\n";
    }

    
    {
       string buf;
       ifstream C2in("Tampon"); 
       ofstream C2out(res.name());
       while(getline(C2in,buf))
           C2out << buf << "\n";
    }
}



int main(int argc,char ** argv)
{

    FileCode FCH(Header);
    FileCode FCElCpp(ElCpp);
    FileCode FCAppliCpp(AppliCpp);

    ElParseDir("../NEW_ELISE/include",FCH);
    ElParseDir("../NEW_ELISE/src",FCElCpp);
    ElParseDir("../NEW_ELISE/applis",FCAppliCpp);
}





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
