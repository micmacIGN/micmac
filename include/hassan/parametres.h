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
#ifndef _HASSAN_PARAMETRES_H
#define _HASSAN_PARAMETRES_H

class Parametres_reconstruction_batiment
{
     public :

        
        Parametres_reconstruction_batiment();
        ostream& operator >>(ostream& os);
        istream& operator <<(istream& is);

        void     set_mode(INT m)              {_mode = m;}
        void     set_afficher(bool a)         {_afficher = a;}
        void     set_enregistrer(bool e)      {_enregistrer = e;}
        void     set_mne_noyau(bool mn)       {_mne_noyau = mn;}

        void     set_repertoir(string& rep)   {_repertoir = rep;} 
        void     set_cad_file(string& cf)     {_cad_file = cf;}
        void     set_mne_file(string& mf)     {_mne_file = mf;}
        void     set_cube_header(string& ch)  {_cube_header = ch;}
        void     set_phot_a(string& pa)       {_phot_a = pa;}
        void     set_phot_b(string& pb)       {_phot_b = pb;}
        void     set_toits_file(string& tf)   {_toits_file = tf;}
        void     set_facades_file(string& ff) {_facades_file = ff;}
        void     set_batis_file(string& bf)   {_batis_file = bf;}

        void     set_sdng(INT sdng)           {_sdng = sdng;}       
        void     set_pas(REAL pxy)            {_pas = pxy;}
        void     set_pas_z(REAL pz)           {_pas_z = pz;}
        void     set_z_sol(REAL zs)           {_z_sol = zs;}

        void     set_seuil_pente(REAL sp)     {_seuil_pente = sp;}
        void     set_seuil_z_min1(REAL szm1)  {_seuil_z_min1 = szm1;} 
        void     set_seuil_z_max1(REAL szm1)  {_seuil_z_max1 = szm1;}
        void     set_seuil_z_min2(REAL szm2)  {_seuil_z_min2 = szm2;}
        void     set_seuil_z_max2(REAL szm2)  {_seuil_z_max2 = szm2;}

        void     set_long_min(REAL lm)        {_long_min = lm;}
        void     set_d_teta_min(REAL dtm)     {_d_teta_min = dtm;}

        void     set_d_teta(REAL dt)          {_d_teta = dt;}
        void     set_d_phi(REAL dp)           {_d_phi  = dp;}
        void     set_d_rho(REAL dr)           {_d_rho  = dr;}
        void     set_nb_max_loc(INT nml)      {_nb_max_loc = nml;}
        void     set_phi_min(REAL pm)         {_phi_min  = pm;}
        void     set_phi_max(REAL pm)         {_phi_max  = pm;}

        void     set_angl_min(REAL am)        {_angl_min = am;}
        void     set_dist_min(REAL dm)        {_dist_min = dm;}
        void     set_decal_max(REAL dm)       {_decal_max = dm;}
        void     set_test_stab(INT ts)        {_test_stab = ts;}                    
        
        void     set_nb_p_mne_min(REAL npmm)  {_nb_p_mne_min = npmm;}
        void     set_nb_p_mne_moy(REAL npmm)  {_nb_p_mne_moy = npmm;}
        void     set_nb_p_mne_max(REAL npmm)  {_nb_p_mne_max = npmm;}
        void     set_nb_p_mne_pas(REAL npmp)  {_nb_p_mne_pas = npmp;}

        void     set_nb_plans_min(INT npm)    {_nb_plans_min = npm;}
        void     set_nb_plans_sup(INT nps)    {_nb_plans_sup = nps;}

        void     set_decal_max_poids_facet_graphe(INT dm){_decal_max_poids_facet_graphe = dm;}
        void     set_test_stab_poids_facet_graphe(INT ts){_test_stab_poids_facet_graphe = ts;}

        void     set_alpha(REAL a)            {_alpha = a;}
        void     set_seuil_sup(REAL ss)       {_seuil_sup = ss;}   
        void     set_seuil_inf(REAL si)       {_seuil_inf = si;}
        void     set_complexite(INT c)        {_complexite = c;}
        void     set_beta1(REAL b1)           {_beta1 = b1;}
        void     set_beta2(REAL b2)           {_beta2 = b2;}
        void     set_nb_sol_gardee(INT nsg)   {_nb_sol_gardee = nsg;}
        void     set_type_correlation(INT tc) {_type_correlation = tc;}
        void     set_prop_file(string pf)     {_prop_file = pf;}
        void     set_resolution_image(bool ri){_resolution_image = ri;}



        INT      mode()            {return _mode;}
        bool     afficher()        {return _afficher;}
        bool     enregistrer()     {return _enregistrer;}
        bool     mne_noyau()       {return _mne_noyau;}

        string   repertoir()       {return _repertoir;}
        string   cad_file()        {return _cad_file;}
        string   mne_file()        {return _mne_file;}
        string   cube_header()     {return _cube_header;}
        string   phot_a()          {return _phot_a;}
        string   phot_b()          {return _phot_b;}
        string   toits_file()      {return _toits_file;}
        string   facades_file()    {return _facades_file;}
        string   batis_file()      {return _batis_file;}

        INT      sdng()            {return _sdng;}
        REAL     pas()             {return _pas;}
        REAL     pas_z()           {return _pas_z;}
        REAL     z_sol()           {return _z_sol;}
        REAL     seuil_pente()     {return _seuil_pente;}
        REAL     seuil_z_min1()    {return _seuil_z_min1;}
        REAL     seuil_z_max1()    {return _seuil_z_max1;}
        REAL     seuil_z_min2()    {return _seuil_z_min2;}
        REAL     seuil_z_max2()    {return _seuil_z_max2;}
        REAL     long_min()        {return _long_min;}
        REAL     d_teta_min()      {return _d_teta_min;}

        REAL     d_teta()          {return _d_teta;}
        REAL     d_phi()           {return _d_phi;}
        REAL     d_rho()           {return _d_rho;}
        INT      nb_max_loc()      {return _nb_max_loc;}
        REAL     phi_min()         {return _phi_min;}
        REAL     phi_max()         {return _phi_max;}

        REAL     angl_min()        {return _angl_min;}
        REAL     dist_min()        {return _dist_min;}
        REAL     decal_max()       {return _decal_max;}
        INT      test_stab()       {return _test_stab;}

        REAL     nb_p_mne_min()  {return _nb_p_mne_min;}
        REAL     nb_p_mne_moy()  {return _nb_p_mne_moy;}
        REAL     nb_p_mne_max()  {return _nb_p_mne_max;}
        REAL     nb_p_mne_pas()  {return _nb_p_mne_pas;}

        INT      nb_plans_min()    {return _nb_plans_min;}
        INT      nb_plans_sup()    {return _nb_plans_sup;}

        INT      decal_max_poids_facet_graphe(){return _decal_max_poids_facet_graphe;}
        INT      test_stab_poids_facet_graphe(){return _test_stab_poids_facet_graphe;}

        REAL     alpha()            {return _alpha;}
        REAL     seuil_sup()        {return _seuil_sup;}
        REAL     seuil_inf()        {return _seuil_inf;}
        INT      complexite()       {return _complexite;}

        REAL     beta1()            {return _beta1;}
        REAL     beta2()            {return _beta2;}

        INT      nb_sol_gardee()    {return _nb_sol_gardee;}
        INT      type_correlation() {return _type_correlation;}

        string   prop_file()        {return _prop_file;}
        bool     resolution_image() {return _resolution_image;}



        INT  _mode;
        bool _afficher;
        bool _enregistrer;
        bool _mne_noyau;

        string _repertoir;
        string _cad_file;
        string _mne_file;
        string _cube_header;
        string _cube_data;
        string _phot_a;
        string _phot_b;
        string _toits_file;
        string _facades_file;
        string _batis_file;

        INT  _sdng;
        REAL _pas;
        REAL _pas_z;
        INT  _cor_fenet;

        REAL _z_sol;           //metre

                                       //parameteres de construir une boite

        REAL _seuil_pente;
        REAL _seuil_z_min1;
        REAL _seuil_z_max1;
        REAL _seuil_z_min2;
        REAL _seuil_z_max2;


        REAL _long_min;               //chercher des directions prinicipales
        REAL _d_teta_min;             //chercher des directions prinicipales 



        REAL _d_teta;                 //transforme de hough
        REAL _d_phi;                  //transforme de hough
        REAL _d_rho;                  //transforme de hough
        INT  _nb_max_loc;             //transforme de hough
        REAL _phi_min;                //transforme de hough
        REAL _phi_max;                //transforme de hough


                                 //filtrage de plans




        REAL _angl_min;               //degree
        REAL _dist_min;               //metre
        REAL _decal_max;              //distance maximale entre un plan et un point de mne
        INT  _test_stab;              //taille d'element de fermeture



        REAL _nb_p_mne_min;           //en sqr metre
        REAL _nb_p_mne_moy;           //en sqr metre
        REAL _nb_p_mne_max;           //en sqr metre
        REAL _nb_p_mne_pas;           //en sqr metre

        INT _nb_plans_min;
        INT _nb_plans_sup;



                                      //pour calculer les poids a partir de MNE


        INT _decal_max_poids_facet_graphe;
        INT _test_stab_poids_facet_graphe;


        REAL _alpha;                    //optimisation noyaux et relaxation
        REAL _seuil_sup;                //le poids minimal no risque
        REAL _seuil_inf;                //le poids minimal no risque
        INT _complexite;                // > 60 temp de calcul est expo

        REAL _beta1;                     //crit�re geo ( fenet horisntale )
        REAL _beta2;                     //crit�re geo ( fenet adaptative )
        INT _nb_sol_gardee;              // n premieres solutions ( fenet hor )
        INT _type_correlation;           //type de correlation : 
                                             //  1 : fenet adap(cent et norm)
                                             //  2 : fenet adap(no cent et norm)
                                             //  3 : corr model(cent et norm)
                                             //  4 : corr model(no cent et norm)
        string _prop_file;
        bool _resolution_image;
};

#endif // _HASSAN_PARAMETRES_H

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
