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
/*eLiSe06/05/99

     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

        eLiSe : Elements of a Linux Image Software Environment

        This program is free software; you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation; either version 2 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to the Free Software
        Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

          Author: Marc PIERROT DESEILLIGNY    IGN/MATIS
          Internet: Marc.Pierrot-Deseilligny@ign.fr
             Phone: (33) 01 43 98 81 28
*/

#ifndef _ELISE_GENERAL_MM_SPEC_ARG_H
#define _ELISE_GENERAL_MM_SPEC_ARG_H

class cMMSpecArg
{
    public :
        // S'agit-il d'un argument optionnel
        bool IsOpt() const;

        // l'argument a-t-il ete initialise
        bool IsInit() const;

        // S'agit-il d'un booleen (ajout pour lever l'ambiguite integer 0/1)
        bool IsBool() const;

        // S'agit-il d'une puissance de 2
        bool IsPowerOf2() const;

        // S'agit-il d'un repertoire
        bool IsDir() const;

        // S'agit-il d'un pattern descriptif de fichier
        bool IsPatFile() const;

        // S'agit-il d'une directory d'orientation existante
        bool IsExistDirOri() const;

        // S'agit-il d'une directory d'orientation en sortie
        bool IsOutputDirOri() const;

        // S'agit-il d'un fichier existant
        bool IsExistFile() const;

        // S'agit-il d'un fichier existant avec un chemin relatif au chantier
        bool IsExistFileWithRelativePath() const;

        // S'agit-il d'un fichier en sortie
        bool IsOutputFile() const;

        // S'agit-il d'une box2d a normaliser
        bool IsToNormalize() const;

        // S'agit-il d'un argument de test
        bool IsForInternalUse() const;

        // Nom du type
        std::string NameType() const;

        // Nom de l'argument (quand optionnel)
        std::string NameArg() const;

        // Commentaire eventuel
        std::string Comment() const;

        // Numero de l'argument dans la specification (pas vraiment utile ??)
        int NumArg() const;

        // Initialise la variable a partir d'une chaine de caractere
        void Init(const std::string &);

        // Liste des valeurs possibles si enumerees, renvoie liste vide sinon
        const std::list<std::string>  & EnumeratedValues() const;

        eArgMainBaseType Type() const;

        template <class T>
        T* DefaultValue() const;

        template <class T>
        bool IsDefaultValue(T val) const;

        cMMSpecArg & operator = (const cMMSpecArg & arg);

private :
        friend class LArgMain;
        cMMSpecArg(GenElArgMain *, int aNum, bool isOpt);

        GenElArgMain * mEAM;
        int            mNum;
        bool           mIsOpt;
};

template <class T>
T* cMMSpecArg::DefaultValue() const { return ( (ElArgMain<T>*)mEAM )->DefVal(); }

template <class T>
bool cMMSpecArg::IsDefaultValue(T val) const { return (IsOpt() && (val == *(DefaultValue<T>()))); }

class cCmpMMSpecArg
{
public :

    cCmpMMSpecArg(){}

    // Comparison; not case sensitive.
    bool operator ()(const cMMSpecArg & aArg0, const cMMSpecArg & aArg1)
    {
        string first  = aArg0.NameArg();
        string second = aArg1.NameArg();

        unsigned int i=0;
        while ((i < first.length()) && (i < second.length()))
        {
            if (tolower (first[i]) < tolower (second[i])) return true;
            else if (tolower (first[i]) > tolower (second[i])) return false;
            i++;
        }

        if (first.length() < second.length()) return true;
        else return false;
    }
};

#endif // _ELISE_GENERAL_MM_SPEC_ARG_H




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant   la mise en
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
associés au chargement,    l'utilisation,    la modification et/ou au
développement et   la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe
manipuler et qui le réserve donc   des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités   charger  et  tester  l'adéquation  du
logiciel   leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder   cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
