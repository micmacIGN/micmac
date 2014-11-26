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



#ifndef _ELISE_PRIVATE_UTI_H
#define _ELISE_PRIVATE_UTI_H



inline INT sub_bit(INT v,INT k0,INT k1)             
        { return (v >> k0) & ((1 << (k1-k0)) -1);}

inline INT set_sub_bit(INT v,INT new_v,INT k0,INT k1)             
{ 
      INT masq = (1 << (k1-k0)) -1;

      return
                (v & ~(masq << k0))  // efface les bits entre k0 et k1
              | ((new_v&masq) << k0);
}

inline INT kth_bit(INT v,INT k)             { return (v & (1<<k)) != 0 ; }

inline INT kth_bit_to_1(INT v, INT k)       { return v | (1<< k)       ; }

inline INT kth_bit_to_0(INT v, INT k)       { return v & (~ (1<< k))   ; }

inline INT set_kth_bit_to_1(INT & v, INT k) { return v |=  1<< k       ; }

inline INT set_kth_bit_to_0(INT & v, INT k) { return v &=  (~ (1<< k)) ; }

inline  INT nb_bits_to_nb_byte(INT nb_bits) { return (nb_bits + 7) / 8; }



inline INT kth_bit_msbf(const U_INT1* v,INT k) 
           { return (v[k/8]&(1<<(7-k%8)))!= 0;}

inline INT kth_bit(const U_INT1* v,INT k) 
           { return (v[k/8]&(1<<(k%8)))!= 0;}

inline U_INT1 kth_bit_to_1(const U_INT1 * v, INT k)  
           { return v[k/8] | (1<< (k%8)) ;}

#if (0)
extern INT sub_bit(INT v,INT k0,INT k1);
INT set_sub_bit(INT v,INT new_v,INT k0,INT k1);
extern INT kth_bit(INT v,INT k);
extern INT kth_bit_to_1(INT v, INT k);
extern INT kth_bit_to_0(INT v, INT k);
extern INT set_kth_bit_to_1(INT & v, INT k);
extern INT set_kth_bit_to_0(INT & v, INT k);
extern  INT nb_bits_to_nb_byte(INT nb_bits);
extern INT kth_bit(const U_INT1* v,INT k) ;
extern INT kth_bit_msbf(const U_INT1* v,INT k) ;
extern U_INT1 kth_bit_to_1(const U_INT1 * v, INT k)  ;
#endif // CPP_OPTIMIZE

INT inv_bits_order(INT val,INT nbb);

void byte_inv_2(void *);  // inverse the byte of a two byte  data
void byte_inv_4(void *);  // inverse the byte of a four byte data
void byte_inv_8(void *);  // inverse the byte of a eight byte data
void byte_inv_16(void *);

void byte_inv_tab(void *,INT byte_by_el,INT nb_el);  
    // inverse the byte of a four byte data


void to_lsb_rep_2(void *);
void to_lsb_rep_4(void *);

void to_msb_rep_2(void *);
void to_msb_rep_4(void *);


extern std::string TheEliseDirXmlSpec;

std::string StdGetFileXMLSpec(const std::string & aName);

// convert upper-case caracter into lower-case
void tolower( std::string &io_str );
std::string tolower( const std::string &i_str);

// convert a filename into a unique representation
// (don't do anything unless under windows because unix's filenames are already unique)
void filename_normalize( std::string &io_filename );
std::string filename_normalize( const std::string &i_str);

// return true if i_str starts with i_start (case sensitive)
bool startWith( const std::string &i_str, const std::string &i_start );

#endif // _ELISE_PRIVATE_UTI_H


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
