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
/*****************************************************************************/
/*                                                                           */
/*  (triangle.h)                                                             */
/*                                                                           */
/*  Include file for programs that call Triangle.                            */
/*                                                                           */
/*  Accompanies Triangle Version 1.3                                         */
/*  July 19, 1996                                                            */
/*                                                                           */
/*  Copyright 1996                                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  How to call Triangle from another program                                */
/*                                                                           */
/*                                                                           */
/*  If you haven't read Triangle's instructions (run "triangle -h" to read   */
/*  them), you won't understand what follows.                                */
/*                                                                           */
/*  Triangle must be compiled into an object file (triangle.o) with the      */
/*  TRILIBRARY symbol defined (preferably by using the -DTRILIBRARY compiler */
/*  switch).  The makefile included with Triangle will do this for you if    */
/*  you run "make trilibrary".  The resulting object file can be called via  */
/*  the procedure triangulate().                                             */
/*                                                                           */
/*  If the size of the object file is important to you, you may wish to      */
/*  generate a reduced version of triangle.o.  The REDUCED symbol gets rid   */
/*  of all features that are primarily of research interest.  Specifically,  */
/*  the -DREDUCED switch eliminates Triangle's -i, -F, -s, and -C switches.  */
/*  The CDT_ONLY symbol gets rid of all meshing algorithms above and beyond  */
/*  constrained Delaunay triangulation.  Specifically, the -DCDT_ONLY switch */
/*  eliminates Triangle's -r, -q, -a, -S, and -s switches.                   */
/*                                                                           */
/*  IMPORTANT:  These definitions (TRILIBRARY, REDUCED, CDT_ONLY) must be    */
/*  made in the makefile or in triangle.c itself.  Putting these definitions */
/*  in this file will not create the desired effect.                         */
/*                                                                           */
/*                                                                           */
/*  The calling convention for triangulate() follows.                        */
/*                                                                           */
/*      void triangulate(triswitches, in, out, vorout)                       */
/*      char *triswitches;                                                   */
/*      struct triangulateio *in;                                            */
/*      struct triangulateio *out;                                           */
/*      struct triangulateio *vorout;                                        */
/*                                                                           */
/*  `triswitches' is a string containing the command line switches you wish  */
/*  to invoke.  No initial dash is required.  Some suggestions:              */
/*                                                                           */
/*  - You'll probably find it convenient to use the `z' switch so that       */
/*    points (and other items) are numbered from zero.  This simplifies      */
/*    indexing, because the first item of any type always starts at index    */
/*    [0] of the corresponding array, whether that item's number is zero or  */
/*    one.                                                                   */
/*  - You'll probably want to use the `Q' (quiet) switch in your final code, */
/*    but you can take advantage of Triangle's printed output (including the */
/*    `V' switch) while debugging.                                           */
/*  - If you are not using the `q' or `a' switches, then the output points   */
/*    will be identical to the input points, except possibly for the         */
/*    boundary markers.  If you don't need the boundary markers, you should  */
/*    use the `N' (no nodes output) switch to save memory.  (If you do need  */
/*    boundary markers, but need to save memory, a good nasty trick is to    */
/*    set out->pointlist equal to in->pointlist before calling triangulate(),*/
/*    so that Triangle overwrites the input points with identical copies.)   */
/*  - The `I' (no iteration numbers) and `g' (.off file output) switches     */
/*    have no effect when Triangle is compiled with TRILIBRARY defined.      */
/*                                                                           */
/*  `in', `out', and `vorout' are descriptions of the input, the output,     */
/*  and the Voronoi output.  If the `v' (Voronoi output) switch is not used, */
/*  `vorout' may be NULL.  `in' and `out' may never be NULL.                 */
/*                                                                           */
/*  Certain fields of the input and output structures must be initialized,   */
/*  as described below.                                                      */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  The `triangulateio' structure.                                           */
/*                                                                           */
/*  Used to pass data into and out of the triangulate() procedure.           */
/*                                                                           */
/*                                                                           */
/*  Arrays are used to store points, triangles, markers, and so forth.  In   */
/*  all cases, the first item in any array is stored starting at index [0].  */
/*  However, that item is item number `1' unless the `z' switch is used, in  */
/*  which case it is item number `0'.  Hence, you may find it easier to      */
/*  index points (and triangles in the neighbor list) if you use the `z'     */
/*  switch.  Unless, of course, you're calling Triangle from a Fortran       */
/*  program.                                                                 */
/*                                                                           */
/*  Description of fields (except the `numberof' fields, which are obvious): */
/*                                                                           */
/*  `pointlist':  An array of point coordinates.  The first point's x        */
/*    coordinate is at index [0] and its y coordinate at index [1], followed */
/*    by the coordinates of the remaining points.  Each point occupies two   */
/*    REALs.                                                                 */
/*  `pointattributelist':  An array of point attributes.  Each point's       */
/*    attributes occupy `numberofpointattributes' REALs.                     */
/*  `pointmarkerlist':  An array of point markers; one int per point.        */
/*                                                                           */
/*  `trianglelist':  An array of triangle corners.  The first triangle's     */
/*    first corner is at index [0], followed by its other two corners in     */
/*    counterclockwise order, followed by any other nodes if the triangle    */
/*    represents a nonlinear element.  Each triangle occupies                */
/*    `numberofcorners' ints.                                                */
/*  `triangleattributelist':  An array of triangle attributes.  Each         */
/*    triangle's attributes occupy `numberoftriangleattributes' REALs.       */
/*  `trianglearealist':  An array of triangle area constraints; one REAL per */
/*    triangle.  Input only.                                                 */
/*  `neighborlist':  An array of triangle neighbors; three ints per          */
/*    triangle.  Output only.                                                */
/*                                                                           */
/*  `segmentlist':  An array of segment endpoints.  The first segment's      */
/*    endpoints are at indices [0] and [1], followed by the remaining        */
/*    segments.  Two ints per segment.                                       */
/*  `segmentmarkerlist':  An array of segment markers; one int per segment.  */
/*                                                                           */
/*  `holelist':  An array of holes.  The first hole's x and y coordinates    */
/*    are at indices [0] and [1], followed by the remaining holes.  Two      */
/*    REALs per hole.  Input only, although the pointer is copied to the     */
/*    output structure for your convenience.                                 */
/*                                                                           */
/*  `regionlist':  An array of regional attributes and area constraints.     */
/*    The first constraint's x and y coordinates are at indices [0] and [1], */
/*    followed by the regional attribute and index [2], followed by the      */
/*    maximum area at index [3], followed by the remaining area constraints. */
/*    Four REALs per area constraint.  Note that each regional attribute is  */
/*    used only if you select the `A' switch, and each area constraint is    */
/*    used only if you select the `a' switch (with no number following), but */
/*    omitting one of these switches does not change the memory layout.      */
/*    Input only, although the pointer is copied to the output structure for */
/*    your convenience.                                                      */
/*                                                                           */
/*  `edgelist':  An array of edge endpoints.  The first edge's endpoints are */
/*    at indices [0] and [1], followed by the remaining edges.  Two ints per */
/*    edge.  Output only.                                                    */
/*  `edgemarkerlist':  An array of edge markers; one int per edge.  Output   */
/*    only.                                                                  */
/*  `normlist':  An array of normal vectors, used for infinite rays in       */
/*    Voronoi diagrams.  The first normal vector's x and y magnitudes are    */
/*    at indices [0] and [1], followed by the remaining vectors.  For each   */
/*    finite edge in a Voronoi diagram, the normal vector written is the     */
/*    zero vector.  Two REALs per edge.  Output only.                        */
/*                                                                           */
/*                                                                           */
/*  Any input fields that Triangle will examine must be initialized.         */
/*  Furthermore, for each output array that Triangle will write to, you      */
/*  must either provide space by setting the appropriate pointer to point    */
/*  to the space you want the data written to, or you must initialize the    */
/*  pointer to NULL, which tells Triangle to allocate space for the results. */
/*  The latter option is preferable, because Triangle always knows exactly   */
/*  how much space to allocate.  The former option is provided mainly for    */
/*  people who need to call Triangle from Fortran code, though it also makes */
/*  possible some nasty space-saving tricks, like writing the output to the  */
/*  same arrays as the input.                                                */
/*                                                                           */
/*  Triangle will not free() any input or output arrays, including those it  */
/*  allocates itself; that's up to you.                                      */
/*                                                                           */
/*  Here's a guide to help you decide which fields you must initialize       */
/*  before you call triangulate().                                           */
/*                                                                           */
/*  `in':                                                                    */
/*                                                                           */
/*    - `pointlist' must always point to a list of points; `numberofpoints'  */
/*      and `numberofpointattributes' must be properly set.                  */
/*      `pointmarkerlist' must either be set to NULL (in which case all      */
/*      markers default to zero), or must point to a list of markers.  If    */
/*      `numberofpointattributes' is not zero, `pointattributelist' must     */
/*      point to a list of point attributes.                                 */
/*    - If the `r' switch is used, `trianglelist' must point to a list of    */
/*      triangles, and `numberoftriangles', `numberofcorners', and           */
/*      `numberoftriangleattributes' must be properly set.  If               */
/*      `numberoftriangleattributes' is not zero, `triangleattributelist'    */
/*      must point to a list of triangle attributes.  If the `a' switch is   */
/*      used (with no number following), `trianglearealist' must point to a  */
/*      list of triangle area constraints.  `neighborlist' may be ignored.   */
/*    - If the `p' switch is used, `segmentlist' must point to a list of     */
/*      segments, `numberofsegments' must be properly set, and               */
/*      `segmentmarkerlist' must either be set to NULL (in which case all    */
/*      markers default to zero), or must point to a list of markers.        */
/*    - If the `p' switch is used without the `r' switch, then               */
/*      `numberofholes' and `numberofregions' must be properly set.  If      */
/*      `numberofholes' is not zero, `holelist' must point to a list of      */
/*      holes.  If `numberofregions' is not zero, `regionlist' must point to */
/*      a list of region constraints.                                        */
/*    - If the `p' switch is used, `holelist', `numberofholes',              */
/*      `regionlist', and `numberofregions' is copied to `out'.  (You can    */
/*      nonetheless get away with not initializing them if the `r' switch is */
/*      used.)                                                               */
/*    - `edgelist', `edgemarkerlist', `normlist', and `numberofedges' may be */
/*      ignored.                                                             */
/*                                                                           */
/*  `out':                                                                   */
/*                                                                           */
/*    - `pointlist' must be initialized (NULL or pointing to memory) unless  */
/*      the `N' switch is used.  `pointmarkerlist' must be initialized       */
/*      unless the `N' or `B' switch is used.  If `N' is not used and        */
/*      `in->numberofpointattributes' is not zero, `pointattributelist' must */
/*      be initialized.                                                      */
/*    - `trianglelist' must be initialized unless the `E' switch is used.    */
/*      `neighborlist' must be initialized if the `n' switch is used.  If    */
/*      the `E' switch is not used and (`in->numberofelementattributes' is   */
/*      not zero or the `A' switch is used), `elementattributelist' must be  */
/*      initialized.  `trianglearealist' may be ignored.                     */
/*    - `segmentlist' must be initialized if the `p' or `c' switch is used,  */
/*      and the `P' switch is not used.  `segmentmarkerlist' must also be    */
/*      initialized under these circumstances unless the `B' switch is used. */
/*    - `edgelist' must be initialized if the `e' switch is used.            */
/*      `edgemarkerlist' must be initialized if the `e' switch is used and   */
/*      the `B' switch is not.                                               */
/*    - `holelist', `regionlist', `normlist', and all scalars may be ignored.*/
/*                                                                           */
/*  `vorout' (only needed if `v' switch is used):                            */
/*                                                                           */
/*    - `pointlist' must be initialized.  If `in->numberofpointattributes'   */
/*      is not zero, `pointattributelist' must be initialized.               */
/*      `pointmarkerlist' may be ignored.                                    */
/*    - `edgelist' and `normlist' must both be initialized.                  */
/*      `edgemarkerlist' may be ignored.                                     */
/*    - Everything else may be ignored.                                      */
/*                                                                           */
/*  After a call to triangulate(), the valid fields of `out' and `vorout'    */
/*  will depend, in an obvious way, on the choice of switches used.  Note    */
/*  that when the `p' switch is used, the pointers `holelist' and            */
/*  `regionlist' are copied from `in' to `out', but no new space is          */
/*  allocated; be careful that you don't free() the same array twice.  On    */
/*  the other hand, Triangle will never copy the `pointlist' pointer (or any */
/*  others); new space is allocated for `out->pointlist', or if the `N'      */
/*  switch is used, `out->pointlist' remains uninitialized.                  */
/*                                                                           */
/*  All of the meaningful `numberof' fields will be properly set; for        */
/*  instance, `numberofedges' will represent the number of edges in the      */
/*  triangulation whether or not the edges were written.  If segments are    */
/*  not used, `numberofsegments' will indicate the number of boundary edges. */
/*                                                                           */
/*****************************************************************************/
 
#ifndef _ELISE_ALGO_GEOM_SHEWCHUCK
#define _ELISE_ALGO_GEOM_SHEWCHUCK      

struct triangulateio {

public :

  triangulateio();
  
  REAL *pointlist;                                               /* In / out */
  REAL *pointattributelist;                                      /* In / out */
  int *pointmarkerlist;                                          /* In / out */
  int numberofpoints;                                            /* In / out */
  int numberofpointattributes;                                   /* In / out */

  int *trianglelist;                                             /* In / out */
  REAL *triangleattributelist;                                   /* In / out */
  REAL *trianglearealist;                                         /* In only */
  int *neighborlist;                                             /* Out only */
  int numberoftriangles;                                         /* In / out */
  int numberofcorners;                                           /* In / out */
  int numberoftriangleattributes;                                /* In / out */

  int *segmentlist;                                              /* In / out */
  int *segmentmarkerlist;                                        /* In / out */
  int numberofsegments;                                          /* In / out */

  REAL *holelist;                        /* In / pointer to array copied out */
  int numberofholes;                                      /* In / copied out */

  REAL *regionlist;                      /* In / pointer to array copied out */
  int numberofregions;                                    /* In / copied out */

  int *edgelist;                                                 /* Out only */
  int *edgemarkerlist;            /* Not used with Voronoi diagram; out only */
  REAL *normlist;                /* Used only with Voronoi diagram; out only */
  int numberofedges;                                             /* Out only */

private :
};

void triangulate
     (
           const char *, 
           struct triangulateio *, 
           struct triangulateio *,
           struct triangulateio *
     ); 

template <class AttrSom,class AttrArc> class  ShewShuckTriangul
{
     public :
         virtual ~ShewShuckTriangul(){}
         ShewShuckTriangul(ElGraphe<AttrSom,AttrArc> &  gr) :
             _gr (gr)
         {
         }

         void triangul
              (
                    ElSubGraphe<AttrSom,AttrArc> &  SubGr,
                    bool                            pslg     = false,
                    bool                            env_conv = true,
                    bool                            voronoi = false
              );
          // FactRandomize : Shewschuck plante sur 3 point alignes.
          // Donc on randomize les coord (sauf si FactRandomize <=0)

     private :

         INT num_shew(const ElSom<AttrSom,AttrArc> & s)
         {
		return _num_shew[s.num()];
         }


          ElGraphe<AttrSom,AttrArc> &      _gr;
          ElFilo<REAL>                     _xy_in;
          ElFilo<REAL>                     _attr_pt_in;
          ElFilo<INT>                      _num_shew;
          ElFilo<INT>                      _pslg_in;
          ElFilo<ElSom<AttrSom,AttrArc> *> _num_to_som;
          virtual void arc_tri
                       (
                              ElSom<AttrSom,AttrArc> *,
                              ElSom<AttrSom,AttrArc> *
                       ) =0;
};

template <class AttrSom,class AttrArc> 
         void ShewShuckTriangul<AttrSom,AttrArc>::triangul
              (
                 ElSubGraphe<AttrSom,AttrArc> &  SubGr,
                 bool                            pslg,
                 bool                            env_conv,
                 bool                            voronoi
              )
{
 // pslg = false;
// env_conv = false;
  
  struct triangulateio in, out,voronoiout;
  string LignCom = "zQ";

  if (env_conv)
     LignCom += "c";

  bool add_arc_tri = true;
  if (add_arc_tri)
     LignCom += "e";

  if (voronoi)
     LignCom += "v";

  if (pslg)
     LignCom += "p";

#if (0)
  steiner = (BOOLEEN)((pslg) && (vc_str_pred(cl_arg,"steiner") != NIL));
  add_predicat_booleen(LignCom,steiner,"s");

  check = (BOOLEEN)(vc_str_pred(cl_arg,"check") != NIL);
  add_predicat_booleen(LignCom,check,"C");

  ang_min = cast_to_reel(vc_str_pred(cl_arg,"ang_min"));
  if (ang_min > 0)
     steiner = Vrai;

  surf_max = cast_to_reel(vc_str_pred(cl_arg,"surf_max"));
  if (surf_max > 0)
     steiner = Vrai;
#endif

  /*************************************************/
  /* Conversion des sommets                        */
  /*************************************************/

    // Pour la numerotation Shewshuck

    while (_gr.maj_num() >= _num_shew.nb())
          _num_shew.pushlast(-1);

    _xy_in.clear();
    _attr_pt_in.clear(); // [1]
    _num_to_som.clear();
    {
       INT NumSom = 0;
       for 
       (
              ElSomIterator<AttrSom,AttrArc> sit = _gr.begin(SubGr) ;
              sit.go_on()                        ;
              sit++         
       )
       {
           Pt2dr pt = SubGr.pt(*sit);
           
           _xy_in.pushlast(pt.x);
           _xy_in.pushlast(pt.y);
           _attr_pt_in.pushlast(pt.x+pt.y);
           _num_shew[(*sit).num()] = NumSom++;
           _num_to_som.pushlast(&(*sit));
       }
    }
    in.numberofpoints =  _xy_in.nb() / 2;
    in.pointlist = _xy_in.tab();
    in.numberofpointattributes = 1;
    in.pointattributelist = _attr_pt_in.tab();


/*
      [1] je ne sais pas pourquoi il faut initaliser 
      "pointattributelist" (d'ailleurs, d'apres la 
      doc ca semble inutile si "numberofpointattributes=0") 
      mais, experimentalement, si on ne le fait pas on se paye une 
      "Error Out of memory".
*/



  /*************************************************/
  /* Lecture des arcs contraints    (pslg)         */
  /*************************************************/

  if(pslg)
  {
       _pslg_in.clear();
       for 
       (
              ElSomIterator<AttrSom,AttrArc> sit = _gr.begin(SubGr) ;
              sit.go_on()                        ;
              sit++         
       )
       {
              ElSom<AttrSom,AttrArc> & s1 = *sit;
              for
              (
                   ElArcIterator<AttrSom,AttrArc> ait = s1.begin(SubGr);
                   ait.go_on();
                   ait++
              )
              {
                   ElSom<AttrSom,AttrArc> & s2 = (*ait).s2();
                   if (s1.num() < s2.num())
                   {
                      _pslg_in.pushlast(num_shew(s1));
                      _pslg_in.pushlast(num_shew(s2));
                   }
              }
       }
       in.segmentlist = _pslg_in.tab();
       in.numberofsegments  = _pslg_in.nb()/2;
  }
  triangulate(LignCom.c_str(), &in, &out, &voronoiout);

#if (0)

  if ((ang_min > 0) || (surf_max > 0))
  {
      struct triangulateio tmp;
     
      sprintf(LignCom,"rzQ");

      if (ang_min > 0)
      {
          sprintf(fin_chaine(LignCom),"q%f",ang_min);
      }
      if (surf_max > 0)
      {
          sprintf(fin_chaine(LignCom),"a%f",surf_max);
      }

      add_predicat_booleen(LignCom,add_arc_tri,"e");

      triangulate(LignCom,&out,&out2,(struct triangulateio Ptr) NULL);
      tmp = out;
      out = out2;
      out2 = tmp;
  }
#endif


#if (0)

  /* Formatage des resultat pour CLISP */

     /* sommet de steiner */

  if (steiner)
  {
     VALEUR Ptr Ptr NSPS,Ptr som;
     int i;

     NSPS = PILE_EVAL+1;
     for (i=0 ; i<nb_som ; i++)
     {
        EMPILER(SPsom[i]);
     }

     for (i=nb_som ; i<out.numberofpoints ; i++)
     {
        som = new_som_gr_qdt_gen
              (  gr,
                 NIL,
                 out.pointlist[i * 2],
                 out.pointlist[i * 2+1],
                 Vrai
              );
        EMPILER(som);
        SET1_FLAG_PRIV(som,flag_steiner);
     }
     nb_som = out.numberofpoints;
     SPsom = NSPS;
  }
#endif


   if (add_arc_tri)
   {
      cout << "Shew " << in.numberofpoints << " => " << out.numberofpoints << "\n";

      for (INT Ke =0 ; Ke<out.numberofedges ; Ke++)
      {
          INT ks1 = out.edgelist[2*Ke];
          INT ks2 = out.edgelist[2*Ke+1];
          if (
                    (ks1 >=0) && (ks2 >=0) 
                 && (ks1<in.numberofpoints) && (ks2<in.numberofpoints)
             )
          {
             
             ElSom<AttrSom,AttrArc> * s1 = _num_to_som[ks1];
             ElSom<AttrSom,AttrArc> * s2 = _num_to_som[ks2];
             if (! SubGr.in(*s1,*s2))
                arc_tri(s1,s2);
          }
          else
          {
/*
               cout << "Shewshuck/MPD, implem incomplete : " 
                    << ks1 << " " << ks2 << " NbPts : " << in.numberofpoints << endl;
*/
               // ELISE_ASSERT(false,"Shewshuck/MPD");
          }
      }
   }


#if (0)
       /* arc de la triangulation */
  if (add_arc_tri)
     shewchuk_add_arc
     (
         out.numberofedges,
         out.edgelist,
         SPsom,
         flag_arc_tri
     );

       /* supression des arcs devenu inutiles par steiner */

  if (autonet_stein)
  {
       int i;
       VALEUR Ptr arc;
       VALEUR Ptr s1,Ptr s2;

       for (i=0 ; i<nb_arc ; i++)
       {
          arc = SParc[i];
          if (! (VAL_FLAG_PRIV(arc,flag_arc_tri)))
          {
              get_s1s2_from_arc(&s1,&s2,arc);
              suppress_arc_gr(s1,s2);
          }
       }
  }


  if (voronoi)
  {
     VALEUR Ptr Ptr SPvor,Ptr som,Ptr dvo;
     QUOD_TREE Ptr qdt;
     DOUBLE x,y;
     int i,k1,k2;
     DOUBLE  xvo,yvo;


     dvo = vc_str_pred(cl_arg,"def_vor_out");
     SPvor = PILE_EVAL+1;
     qdt = grvor->QDT_GR->QDT;
     cast_to_complexe_reel(dvo,&xvo,&yvo);
     for (i=0 ; i<vorout.numberofpoints ; i++)
     {
        x =  vorout.pointlist[i * 2];
        y =  vorout.pointlist[i * 2 + 1];
        if ( 
                (qdt->deb.x < x)
             && (x < qdt->fin.x)
             && (qdt->deb.y < y)
             && (y < qdt->fin.y)
           )
        {
            EMPILER(NIL);
        }
        else
        {
            EMPILER(allouer_pt2d_reel(x,y));
            x = xvo;
            y = yvo;
        }
        som = new_som_gr_qdt_gen
              (  grvor,
                 PILE_EVAL[0],
                 x,
                 y,
                 Vrai
              );
        PILE_EVAL--;
        EMPILER(som);
     }
     shewchuk_add_arc
     (
         vorout.numberofedges,
         vorout.edgelist,
         SPvor,
         -1
     );




     {
         VALEUR Ptr s1,Ptr s2,Ptr arc;
         for (i=0 ; i<vorout.numberofedges ; i++)
         {
            k1 = vorout.edgelist[i*2];
            k2 = vorout.edgelist[i*2+1];
            if (k2==-1)
            {
                s1 = SPvor[k1];
                if (s1->ATTR_S_GR != NIL)
                   cast_to_complexe_reel(s1->ATTR_S_GR,&x,&y);
                else
                   cast_to_complexe_reel(s1,&x,&y);
               
                EMPILER(allouer_pt2d_reel(x+vorout.normlist[2*i],y+vorout.normlist[2*i+1]));
                s2 = new_som_gr_qdt_gen
                      (  grvor,
                         PILE_EVAL[0],
                         xvo,
                         yvo,
                         Vrai
                      );
               PILE_EVAL--;

               SET1_FLAG_PRIV(s2,flag_inf_vor);
               arc = arc_s1s2_or_create(s1,s2,0.0);
               SET1_FLAG_PRIV(arc,flag_inf_vor);
               SET1_FLAG_PRIV(arc_rec(arc),flag_inf_vor);
            }
          }
      }
  }


  liberer_arg_chem_gr(&arg_ch);

  PILE_EVAL = SP0;
#endif
}



#if (0)

void triangulate(char *, struct triangulateio *, struct triangulateio *,
                 struct triangulateio *);
#include "def_type.h"
#include "fonction_privees.h"
#include "var_glob.h"
#include "ges_tables.h"
#include "qdt.h"
#include "new_gr_cl.h"
#include "X11_include.h"


/*****************************************************************************/
/*                                                                           */
/*  (tricall.c)                                                              */
/*                                                                           */
/*  Example program that demonstrates how to call Triangle.                  */
/*                                                                           */
/*  Accompanies Triangle Version 1.3                                         */
/*  July 19, 1996                                                            */
/*                                                                           */
/*  This file is placed in the public domain (but the file that it calls     */
/*  is still copyrighted!) by                                                */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*****************************************************************************/

/* If SINGLE is defined when triangle.o is compiled, it should also be       */
/*   defined here.  If not, it should not be defined here.                   */

/* #define SINGLE */

#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */

#include <cstdio>
#include <cstdlib>
#include "triangle.h"

#ifndef _STDLIB_H_
extern void *malloc();
extern void free();
#endif /* _STDLIB_H_ */

/*****************************************************************************/
/*                                                                           */
/*  report()   Print the input or output.                                    */
/*                                                                           */
/*****************************************************************************/

/*
void report(io, markers, reporttriangles, reportneighbors, reportsegments,
            reportedges, reportnorms)
*/
void report(
        struct triangulateio *io,
        int markers,
        int reporttriangles,
        int reportneighbors,
        int reportsegments,
        int reportedges,
        int reportnorms 
    )
{
  int i, j;


  for (i = 0; i < io->numberofpoints; i++) {
    printf("Point %4d:", i);
    for (j = 0; j < 2; j++) {
      printf("  %.6g", io->pointlist[i * 2 + j]);
    }
    if (io->numberofpointattributes > 0) {
      printf("   attributes");
    }
    for (j = 0; j < io->numberofpointattributes; j++) {
      printf("  %.6g",
             io->pointattributelist[i * io->numberofpointattributes + j]);
    }
    if (markers && io->pointmarkerlist) {
      printf("   marker %d\n", io->pointmarkerlist[i]);
    } else {
      printf("\n");
    }
  }
  printf("\n");

  if (io->trianglelist && io->triangleattributelist && (reporttriangles || reportneighbors)) 
  {
    for (i = 0; i < io->numberoftriangles; i++) {
      if (reporttriangles) {
        printf("Triangle %4d points:", i);
        for (j = 0; j < io->numberofcorners; j++) {
          printf("  %4d", io->trianglelist[i * io->numberofcorners + j]);
        }
        if (io->numberoftriangleattributes > 0) {
          printf("   attributes");
        }
        for (j = 0; j < io->numberoftriangleattributes; j++) {
          printf("  %.6g", io->triangleattributelist[i *
                                         io->numberoftriangleattributes + j]);
        }
        printf("\n");
      }
      if (reportneighbors) {
        printf("Triangle %4d neighbors:", i);
        for (j = 0; j < 3; j++) {
          printf("  %4d", io->neighborlist[i * 3 + j]);
        }
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportsegments && io->segmentlist) {
    for (i = 0; i < io->numberofsegments; i++) {
      printf("Segment %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->segmentlist[i * 2 + j]);
      }
      if (markers) {
        printf("   marker %d\n", io->segmentmarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportedges && io->edgelist) {
    for (i = 0; i < io->numberofedges; i++) {
      printf("Edge %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->edgelist[i * 2 + j]);
      }
      if (reportnorms && (io->edgelist[i * 2 + 1] == -1)) {
        for (j = 0; j < 2; j++) {
          printf("  %.6g", io->normlist[i * 2 + j]);
        }
      }
      if (markers) {
        printf("   marker %d\n", io->edgemarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }
}

/*****************************************************************************/
/*                                                                           */
/*  main()   Create and refine a mesh.                                       */
/*                                                                           */
/*****************************************************************************/

int trbid(void)
{
  struct triangulateio in, mid, out, vorout;

  /* Define input points. */

  in.numberofpoints = 4;
  in.numberofpointattributes = 1;
  in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
  in.pointlist[0] = 0.0;
  in.pointlist[1] = 0.0;
  in.pointlist[2] = 1.0;
  in.pointlist[3] = 0.0;
  in.pointlist[4] = 1.0;
  in.pointlist[5] = 9.0;
  in.pointlist[6] = 0.0;
  in.pointlist[7] = 10.0;
  in.pointattributelist = (REAL *) malloc(in.numberofpoints *
                                          in.numberofpointattributes *
                                          sizeof(REAL));
  in.pointattributelist[0] = 0.0;
  in.pointattributelist[1] = 1.0;
  in.pointattributelist[2] = 11.0;
  in.pointattributelist[3] = 10.0;



  in.pointmarkerlist = (int *) NULL;
/*
  in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
  in.pointmarkerlist[0] = 0;
  in.pointmarkerlist[1] = 2;
  in.pointmarkerlist[2] = 0;
  in.pointmarkerlist[3] = 0;
*/


#if (0)
  in.numberofsegments = 0;
  in.segmentlist = (int *) NULL;
#endif
  in.numberofsegments = 1;
  in.segmentlist = (int *) malloc(2* in.numberofsegments * sizeof(int));
  in.segmentlist[0] = 1;
  in.segmentlist[1] = 3;




  in.segmentmarkerlist = (int *) NULL;



  in.numberofholes = 0;

  in.numberofregions = 0;
  in.regionlist = (REAL *) NULL;;
#if(0)
  in.numberofregions = 1;
  in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));
  in.regionlist[0] = 0.5;
  in.regionlist[1] = 5.0;
  in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
  in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */
#endif

  printf("------------------------ Input point set ------------------------\n");
  report(&in, 1, 0, 0, 0, 0, 0);
  getchar(); 

  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

  mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  mid.pointattributelist = (REAL *) NULL;
  mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
  mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  mid.triangleattributelist = (REAL *) NULL;
  mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  mid.segmentlist = (int *) NULL;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  mid.segmentmarkerlist = (int *) NULL;
  mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
  mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

  vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
  /* Needed only if -v switch used and number of attributes is not zero: */
  vorout.pointattributelist = (REAL *) NULL;
  vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
  vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

  /* Triangulate the points.  Switches are chosen to read and write a  */
  /*   PSLG (p), preserve the convex hull (c), number everything from  */
  /*   zero (z), assign a regional attribute to each element (A), and  */
  /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
  /*   neighbor list (n).                                              */

  triangulate("pczAevnQ", &in, &mid, &vorout);

  printf("------------------------  Initial triangulation  ------------------------\n");
  report(&mid, 1, 1, 1, 1, 1, 0);
  getchar();
  /* Initial Voronoi diagram */
  printf("------------------------  Voronoi out  ------------------------\n");
  report(&vorout, 0, 0, 0, 0, 1, 1);
  getchar();

  /* Attach area constraints to the triangles in preparation for */
  /*   refining the triangulation.                               */

  /* Needed only if -r and -a switches used: */
  mid.trianglearealist = (REAL *) malloc(mid.numberoftriangles * sizeof(REAL));
  mid.trianglearealist[0] = 3.0;
  mid.trianglearealist[1] = 1.0;

  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `out'.                                    */

  out.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of attributes is zero: */
  out.pointattributelist = (REAL *) NULL;
  out.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  out.triangleattributelist = (REAL *) NULL;

  /* Refine the triangulation according to the attached */
  /*   triangle area constraints.                       */

  triangulate("q20przBPQ", &mid, &out, (struct triangulateio *) NULL);

  printf("------------------------  Refined out  ------------------------\n");
  report(&out, 0, 1, 0, 0, 0, 0);
  getchar();

  /* Free all allocated arrays, including those allocated by Triangle. */

  free(in.pointlist);
  free(in.pointattributelist);
  free(in.pointmarkerlist);
  free(in.regionlist);
  free(mid.pointlist);
  free(mid.pointattributelist);
  free(mid.pointmarkerlist);
  free(mid.trianglelist);
  free(mid.triangleattributelist);
  free(mid.trianglearealist);
  free(mid.neighborlist);
  free(mid.segmentlist);
  free(mid.segmentmarkerlist);
  free(mid.edgelist);
  free(mid.edgemarkerlist);
  free(vorout.pointlist);
  free(vorout.pointattributelist);
  free(vorout.edgelist);
  free(vorout.normlist);
  free(out.pointlist);
  free(out.pointattributelist);
  free(out.trianglelist);
  free(out.triangleattributelist);

  return 0;
}


struct triangulateio init_trio_shew(void)
{
     struct triangulateio res;

     res.pointlist                   = (REAL *) NULL;
     res.pointattributelist          = (REAL *) NULL;
     res.pointmarkerlist             = (int *) NULL;
     res.numberofpoints              = 0;
     res.numberofpointattributes     = 0;
    
     res.trianglelist                = (int *)  NULL;
     res.triangleattributelist       = (REAL *) NULL;
     res.trianglearealist            = (REAL *) NULL;
     res.neighborlist                = (int  *) NULL;
     res.numberoftriangles           = 0;
     res.numberofcorners             = 0;
     res.numberoftriangleattributes  = 0;

     res.segmentlist                 = (int *) NULL;
     res.segmentmarkerlist           = (int *) NULL;
     res.numberofsegments            = 0;
     
     res.holelist                    = (REAL *) NULL;
     res.numberofholes               = 0;

     res.regionlist                  = (REAL *) NULL;
     res.numberofregions             = 0;


     res.edgelist                    = (int *) NULL;
     res.edgemarkerlist              = (int *) NULL;
     res.normlist                    = (REAL *) NULL;
     res.numberofedges               = 0;

     return res;
}

char * fin_chaine(char * ch)
{
   return ch + strlen(ch);
}

void add_predicat_booleen(char * ch,BOOLEEN test,CONST char * opt)
{
   if (test)
      sprintf(fin_chaine(ch),opt);
}



void shewchuk_add_arc FONC_4
     (
          INT              ,nb,
          INT Ptr          ,tarc,
          VALEUR Ptr Ptr   ,tabsom,
          INT              ,flag
     )
{
       INT i,k1,k2;
       VALEUR Ptr s1,Ptr s2,Ptr arc;

       for (i=0 ; i<nb ; i++)
       {
          k1 = tarc[i*2];
          k2 = tarc[i*2+1];
          if ((k1>=0)&&(k2>=0))
          {
              s1 = tabsom[k1];
              s2 = tabsom[k2];
              arc = arc_s1s2_or_create(s1,s2,0.0);
              if (flag >= 0)
              {
                  SET1_FLAG_PRIV(arc,flag);
                  SET1_FLAG_PRIV(arc_rec(arc),flag);
              }
          }
      }
}






void  trmain FONC_2 (VALEUR Ptr,gr,VALEUR Ptr,cl_arg)
{
  struct triangulateio in, out, vorout,out2;
  VALEUR Ptr Ptr SPsom,Ptr Ptr SP0,Ptr Ptr SParc;
  VALEUR Ptr grvor;
  ARG_CHEM_GR arg_ch;
  INT nb_som,nb_arc;
  char  LignCom[300];
  INT Ptr etat;
  BOOLEEN env_conv,add_arc_tri,debug,pslg,voronoi,steiner,autonet_stein,check;
  INT     flag_arc_tri,flag_steiner,flag_inf_vor;
  DOUBLE  ang_min,surf_max;


  /* gcc -O -Wall */
  SParc = (VALEUR **) NULL;
  nb_arc = 12345678;

  etat = ENREGISTER_ETAT_BANQUE();
  sprintf(LignCom,"zQ");  /*Anv*/
  SP0 = PILE_EVAL;

  in     = init_trio_shew();
  out    = init_trio_shew();
  out2   = init_trio_shew();
  vorout = init_trio_shew();

   /************************/
  /* lecture des arguments */

  arg_ch = lire_arg_chem_gr(vc_str_pred(cl_arg,"ss_gr"),gr,0);

  env_conv = (BOOLEEN)(vc_str_pred(cl_arg,"env_conv") != NIL);
  add_predicat_booleen(LignCom,env_conv,"c");

  add_arc_tri = (BOOLEEN)(vc_str_pred(cl_arg,"add_arc_tri") != NIL);
  add_predicat_booleen(LignCom,add_arc_tri,"e");

  voronoi = (BOOLEEN)((grvor = vc_str_pred(cl_arg,"voronoi")) != NIL);
  add_predicat_booleen(LignCom,voronoi,"v");

  pslg = (BOOLEEN)((!voronoi) && (vc_str_pred(cl_arg,"pslg") != NIL));
  add_predicat_booleen(LignCom,pslg,"p");

  steiner = (BOOLEEN)((pslg) && (vc_str_pred(cl_arg,"steiner") != NIL));
  add_predicat_booleen(LignCom,steiner,"s");

  check = (BOOLEEN)(vc_str_pred(cl_arg,"check") != NIL);
  add_predicat_booleen(LignCom,check,"C");


  ang_min = cast_to_reel(vc_str_pred(cl_arg,"ang_min"));
  if (ang_min > 0)
     steiner = Vrai;

  surf_max = cast_to_reel(vc_str_pred(cl_arg,"surf_max"));
  if (surf_max > 0)
     steiner = Vrai;

  autonet_stein = (BOOLEEN)((steiner) && (vc_str_pred(cl_arg,"autonet_stein") != NIL));

  debug =  (BOOLEEN) (vc_str_pred(cl_arg,"debug") != NIL);


  flag_arc_tri  = (vc_str_pred(cl_arg,"flag_arc_tri") )->NFL_EXFL +DEB_FLAG_USER_GR;

  flag_steiner  = (vc_str_pred(cl_arg,"flag_steiner") )->NFL_EXFL +DEB_FLAG_USER_GR;

  flag_inf_vor  = (vc_str_pred(cl_arg,"flag_inf_vor") )->NFL_EXFL +DEB_FLAG_USER_GR;

  /**********************************************************************************/
  /* lecture des sommets du graphes . Rqs :

      [1] je ne sais pas pourquoi il faut initaliser "pointattributelist" (d'ailleurs,
          d'apres la doc ca semble inutile si "numberofpointattributes=0") mais,
          experimentalement, si on ne le fait pas on se paye une "Error Out of memory".
    
  */
  /**********************************************************************************/

  SPsom = PILE_EVAL +1;

  nb_som = empiler_all_soms(gr,&arg_ch);


  in.numberofpoints = nb_som;
  in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));

  in.numberofpointattributes = 1;
  in.pointattributelist = (REAL *) malloc(in.numberofpoints * 1 * sizeof(REAL));
  {
     int i;
     DOUBLE x,y;
     for (i=0 ; i<nb_som ; i++)
     {
          SPsom[i]->PERE_S_GR = i;
          cast_to_complexe_reel(SPsom[i],&x,&y);
          in.pointlist[2*i]   = x;
          in.pointlist[2*i+1] = y;
          in.pointattributelist[i] = x + y;
     }
  }


  /*************************************************/
  /* Lecture des arcs contraints    (pslg)         */
  /*************************************************/

  if(pslg)
  {
      VALEUR Ptr s1,Ptr s2,Ptr arc;
      INT i;

      SParc = PILE_EVAL +1;   
      nb_arc = empiler_all_arcs(gr,&arg_ch,Faux);
      in.numberofsegments = nb_arc;
      in.segmentlist = (int *) malloc(2* in.numberofsegments * sizeof(int));
      for (i = 0 ; i<nb_arc ; i++)
      {
          arc = SParc[i];
          SET0_FLAG_PRIV(arc,flag_arc_tri);
          SET0_FLAG_PRIV(arc_rec(arc),flag_arc_tri);
          get_s1s2_from_arc(&s1,&s2,arc);
          in.segmentlist[2*i] =   s1->PERE_S_GR;
          in.segmentlist[2*i+1] = s2->PERE_S_GR;
      }
  }



  if (debug)
  {
       printf("LIGNE DE COMMANDE : [%s]\n",LignCom);
       printf("------------------------ Input point set ------------------------\n");
       report(&in, 1, 0, 0, 0, 0, 0);
       getchar();
  }


  /* triangulate("pczAevnQ", &in, &mid, &vorout); */

  triangulate(LignCom, &in, &out, &vorout);

  if ((ang_min > 0) || (surf_max > 0))
  {
      struct triangulateio tmp;
     
      sprintf(LignCom,"rzQ");

      if (ang_min > 0)
      {
          sprintf(fin_chaine(LignCom),"q%f",ang_min);
      }
      if (surf_max > 0)
      {
          sprintf(fin_chaine(LignCom),"a%f",surf_max);
      }

      add_predicat_booleen(LignCom,add_arc_tri,"e");

      triangulate(LignCom,&out,&out2,(struct triangulateio Ptr) NULL);
      tmp = out;
      out = out2;
      out2 = tmp;
  }

  if (debug)
  {
      printf("------------------------  Initial triangulation  ------------------------\n");
      report(&out, 1, 1, 1, 1, 1, 0);
      getchar();
  }



  /* Formatage des resultat pour CLISP */


     /* sommet de steiner */

  if (steiner)
  {
     VALEUR Ptr Ptr NSPS,Ptr som;
     int i;

     NSPS = PILE_EVAL+1;
     for (i=0 ; i<nb_som ; i++)
     {
        EMPILER(SPsom[i]);
     }

     for (i=nb_som ; i<out.numberofpoints ; i++)
     {
        som = new_som_gr_qdt_gen
              (  gr,
                 NIL,
                 out.pointlist[i * 2],
                 out.pointlist[i * 2+1],
                 Vrai
              );
        EMPILER(som);
        SET1_FLAG_PRIV(som,flag_steiner);
     }
     nb_som = out.numberofpoints;
     SPsom = NSPS;
  }

       /* arc de la triangulation */
  if (add_arc_tri)
     shewchuk_add_arc
     (
         out.numberofedges,
         out.edgelist,
         SPsom,
         flag_arc_tri
     );

       /* supression des arcs devenu inutiles par steiner */

  if (autonet_stein)
  {
       int i;
       VALEUR Ptr arc;
       VALEUR Ptr s1,Ptr s2;

       for (i=0 ; i<nb_arc ; i++)
       {
          arc = SParc[i];
          if (! (VAL_FLAG_PRIV(arc,flag_arc_tri)))
          {
              get_s1s2_from_arc(&s1,&s2,arc);
              suppress_arc_gr(s1,s2);
          }
       }
  }


  if (voronoi)
  {
     VALEUR Ptr Ptr SPvor,Ptr som,Ptr dvo;
     QUOD_TREE Ptr qdt;
     DOUBLE x,y;
     int i,k1,k2;
     DOUBLE  xvo,yvo;


     dvo = vc_str_pred(cl_arg,"def_vor_out");
     SPvor = PILE_EVAL+1;
     qdt = grvor->QDT_GR->QDT;
     cast_to_complexe_reel(dvo,&xvo,&yvo);
     for (i=0 ; i<vorout.numberofpoints ; i++)
     {
        x =  vorout.pointlist[i * 2];
        y =  vorout.pointlist[i * 2 + 1];
        if ( 
                (qdt->deb.x < x)
             && (x < qdt->fin.x)
             && (qdt->deb.y < y)
             && (y < qdt->fin.y)
           )
        {
            EMPILER(NIL);
        }
        else
        {
            EMPILER(allouer_pt2d_reel(x,y));
            x = xvo;
            y = yvo;
        }
        som = new_som_gr_qdt_gen
              (  grvor,
                 PILE_EVAL[0],
                 x,
                 y,
                 Vrai
              );
        PILE_EVAL--;
        EMPILER(som);
     }
     shewchuk_add_arc
     (
         vorout.numberofedges,
         vorout.edgelist,
         SPvor,
         -1
     );




     {
         VALEUR Ptr s1,Ptr s2,Ptr arc;
         for (i=0 ; i<vorout.numberofedges ; i++)
         {
            k1 = vorout.edgelist[i*2];
            k2 = vorout.edgelist[i*2+1];
            if (k2==-1)
            {
                s1 = SPvor[k1];
                if (s1->ATTR_S_GR != NIL)
                   cast_to_complexe_reel(s1->ATTR_S_GR,&x,&y);
                else
                   cast_to_complexe_reel(s1,&x,&y);
               
                EMPILER(allouer_pt2d_reel(x+vorout.normlist[2*i],y+vorout.normlist[2*i+1]));
                s2 = new_som_gr_qdt_gen
                      (  grvor,
                         PILE_EVAL[0],
                         xvo,
                         yvo,
                         Vrai
                      );
               PILE_EVAL--;

               SET1_FLAG_PRIV(s2,flag_inf_vor);
               arc = arc_s1s2_or_create(s1,s2,0.0);
               SET1_FLAG_PRIV(arc,flag_inf_vor);
               SET1_FLAG_PRIV(arc_rec(arc),flag_inf_vor);
            }
          }
      }
  }


  liberer_arg_chem_gr(&arg_ch);

  PILE_EVAL = SP0;
  return;




#if(0)
  /* Define input points. */

  in.numberofpoints = 4;
  in.numberofpointattributes = 1;
  in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
  in.pointlist[0] = 0.0;
  in.pointlist[1] = 0.0;
  in.pointlist[2] = 1.0;
  in.pointlist[3] = 0.0;
  in.pointlist[4] = 1.0;
  in.pointlist[5] = 9.0;
  in.pointlist[6] = 0.0;
  in.pointlist[7] = 10.0;
  in.pointattributelist = (REAL *) malloc(in.numberofpoints *
                                          in.numberofpointattributes *
                                          sizeof(REAL));
  in.pointattributelist[0] = 0.0;
  in.pointattributelist[1] = 1.0;
  in.pointattributelist[2] = 11.0;
  in.pointattributelist[3] = 10.0;
  in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
  in.pointmarkerlist[0] = 0;
  in.pointmarkerlist[1] = 2;
  in.pointmarkerlist[2] = 0;
  in.pointmarkerlist[3] = 0;


#if (0)
  in.numberofsegments = 0;
  in.segmentlist = (int *) NULL;
#endif
  in.numberofsegments = 1;
  in.segmentlist = (int *) malloc(2* in.numberofsegments * sizeof(int));
  in.segmentlist[0] = 1;
  in.segmentlist[1] = 3;
  in.segmentmarkerlist = (int *) NULL;



  in.numberofholes = 0;

  in.numberofregions = 0;
  in.regionlist = (REAL *) NULL;;
#if(0)
  in.numberofregions = 1;
  in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));
  in.regionlist[0] = 0.5;
  in.regionlist[1] = 5.0;
  in.regionlist[2] = 7.0;            /* Regional attribute (for whole mesh). */
  in.regionlist[3] = 0.1;          /* Area constraint that will not be used. */
#endif

  printf("------------------------ Input point set ------------------------\n");
  report(&in, 1, 0, 0, 0, 0, 0);
  getchar(); 

  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

  mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  mid.pointattributelist = (REAL *) NULL;
  mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
  mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  mid.triangleattributelist = (REAL *) NULL;
  mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  mid.segmentlist = (int *) NULL;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  mid.segmentmarkerlist = (int *) NULL;
  mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
  mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

  vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
  /* Needed only if -v switch used and number of attributes is not zero: */
  vorout.pointattributelist = (REAL *) NULL;
  vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
  vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

  /* Triangulate the points.  Switches are chosen to read and write a  */
  /*   PSLG (p), preserve the convex hull (c), number everything from  */
  /*   zero (z), assign a regional attribute to each element (A), and  */
  /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
  /*   neighbor list (n).                                              */

  triangulate("spczAevnQ", &in, &mid, &vorout);

  printf("------------------------  Initial triangulation  ------------------------\n");
  report(&mid, 1, 1, 1, 1, 1, 0);
  getchar();
  /* Initial Voronoi diagram */
  printf("------------------------  Voronoi out  ------------------------\n");
  report(&vorout, 0, 0, 0, 0, 1, 1);
  getchar();

  /* Attach area constraints to the triangles in preparation for */
  /*   refining the triangulation.                               */

  /* Needed only if -r and -a switches used: */
  mid.trianglearealist = (REAL *) malloc(mid.numberoftriangles * sizeof(REAL));
  mid.trianglearealist[0] = 3.0;
  mid.trianglearealist[1] = 1.0;

  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `out'.                                    */

  out.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of attributes is zero: */
  out.pointattributelist = (REAL *) NULL;
  out.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  out.triangleattributelist = (REAL *) NULL;

  /* Refine the triangulation according to the attached */
  /*   triangle area constraints.                       */

  triangulate("prazBPQ", &mid, &out, (struct triangulateio *) NULL);

  printf("------------------------  Refined out  ------------------------\n");
  report(&out, 0, 1, 0, 0, 0, 0);
  getchar();

  /* Free all allocated arrays, including those allocated by Triangle. */

  free(in.pointlist);
  free(in.pointattributelist);
  free(in.pointmarkerlist);
  free(in.regionlist);
  free(mid.pointlist);
  free(mid.pointattributelist);
  free(mid.pointmarkerlist);
  free(mid.trianglelist);
  free(mid.triangleattributelist);
  free(mid.trianglearealist);
  free(mid.neighborlist);
  free(mid.segmentlist);
  free(mid.segmentmarkerlist);
  free(mid.edgelist);
  free(mid.edgemarkerlist);
  free(vorout.pointlist);
  free(vorout.pointattributelist);
  free(vorout.edgelist);
  free(vorout.normlist);
  free(out.pointlist);
  free(out.pointattributelist);
  free(out.trianglelist);
  free(out.triangleattributelist);
#endif

}

void fonction_shewchuk()
{ 
    if (NB_ARG == 0)
       trbid();
    else
       trmain(PILE_EVAL[1-NB_ARG],PILE_EVAL[2-NB_ARG]);

   PILE_EVAL -= NB_ARG;
}
#endif
#endif // _ELISE_ALGO_GEOM_SHEWCHUCK      

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
