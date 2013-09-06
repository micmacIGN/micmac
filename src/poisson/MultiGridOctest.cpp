/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "StdAfx.h"

char* outputFile=NULL;
int echoStdout=0;
void DumpOutput( const char* format , ... )
{
	if( outputFile )
	{
		FILE* fp = fopen( outputFile , "a" );
		va_list args;
		va_start( args , format );
		vfprintf( fp , format , args );
		fclose( fp );
		va_end( args );
	}
	if( echoStdout )
	{
		va_list args;
		va_start( args , format );
		vprintf( format , args );
		va_end( args );
	}
}
void DumpOutput2( char* str , const char* format , ... )
{
	if( outputFile )
	{
		FILE* fp = fopen( outputFile , "a" );
		va_list args;
		va_start( args , format );
		vfprintf( fp , format , args );
		fclose( fp );
		va_end( args );
	}
	if( echoStdout )
	{
		va_list args;
		va_start( args , format );
		vprintf( format , args );
		va_end( args );
	}
	va_list args;
	va_start( args , format );
	vsprintf( str , format , args );
	va_end( args );
	if( str[strlen(str)-1]=='\n' ) str[strlen(str)-1] = 0;
}

#include "poisson/MultiGridOctreeData.h"

cmdLineString
	In( "in" ) ,
	Out( "out" ) ,
	VoxelGrid( "voxel" );

cmdLineReadable
#ifdef _WIN32
	Performance( "performance" ) ,
#endif // _WIN32
	Verbose( "verbose" ) ,
	NoComments( "noComments" ) ,
	PolygonMesh( "polygonMesh" ) ,
	NonAdaptiveWeights( "nonAdaptive" ) ,
	Confidence( "confidence" ) ,
	NonManifold( "nonManifold" ) ,
	ASCII( "ascii" ) ,
	ShowResidual( "showResidual" );

cmdLineInt
	Depth( "depth" , 8 ) ,
	SolverDivide( "solverDivide" , 8 ) ,
	IsoDivide( "isoDivide" , 8 ) ,
	KernelDepth( "kernelDepth" ) ,
	MinIters( "minIters" , 8 ) ,
	VoxelDepth( "voxelDepth" , -1 ) ,
	MinDepth( "minDepth" , 5 ) ,
	Threads( "threads" , 1 );
	//TODO: Threads( "threads" , omp_get_num_procs() );

cmdLineFloat
	SamplesPerNode( "samplesPerNode" , 1.f ) ,
	Scale( "scale" , 1.1f ) ,
	SolverAccuracy( "accuracy" , float(1e-3) ) ,
	PointWeight( "pointWeight" , 4.f );


cmdLineReadable* params[] =
{
	&In , &Depth , &Out ,
	&SolverDivide , &IsoDivide , &Scale , &Verbose , &SolverAccuracy , &NoComments ,
	&KernelDepth , &SamplesPerNode , &Confidence , &NonManifold , &PolygonMesh , &ASCII , &ShowResidual , &MinIters , &NonAdaptiveWeights , &VoxelDepth ,
	&PointWeight , &VoxelGrid , &Threads , &MinDepth ,
#ifdef _WIN32
	&Performance ,
#endif // _WIN32
};


void ShowUsage(char* ex)
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s  <input points>\n" , In.name );

	printf( "\t[--%s <ouput triangle mesh>]\n" , Out.name );
	printf( "\t[--%s <ouput voxel grid>]\n" , VoxelGrid.name );

	printf( "\t[--%s <maximum reconstruction depth>=%d]\n" , Depth.name , Depth.value );
	printf( "\t\t Running at depth d corresponds to solving on a 2^d x 2^d x 2^d\n" );
	printf( "\t\t voxel grid.\n" );

	printf( "\t[--%s <depth at which to extract the voxel grid>=<%s>]\n" , VoxelDepth.name , Depth.name );

	printf( "\t[--%s <scale factor>=%f]\n" , Scale.name , Scale.value );
	printf( "\t\t Specifies the factor of the bounding cube that the input\n" );
	printf( "\t\t samples should fit into.\n" );

	printf( "\t[--%s <subdivision depth>=%d]\n" , SolverDivide.name , SolverDivide.value );
	printf( "\t\t The depth at which a block Gauss-Seidel solver is used\n");
	printf( "\t\t to solve the Laplacian.\n");

	printf( "\t[--%s <minimum number of samples per node>=%f]\n" , SamplesPerNode.name, SamplesPerNode.value );
	printf( "\t\t This parameter specifies the minimum number of points that\n" );
	printf( "\t\t should fall within an octree node.\n" );

	printf( "\t[--%s <num threads>=%d]\n" , Threads.name , Threads.value );
	printf( "\t\t This parameter specifies the number of threads across which\n" );
	printf( "\t\t the solver should be parallelizeds.\n" );

	printf( "\t[--%s]\n" , Confidence.name );
	printf( "\t\t If this flag is enabled, the size of a sample's normals is\n" );
	printf( "\t\t used as a confidence value, affecting the sample's\n" );
	printf( "\t\t constribution to the reconstruction process.\n" );

	printf( "\t[--%s]\n" , NonManifold.name );
	printf( "\t\t If this flag is enabled, the isosurface extraction does not add\n" );
	printf( "\t\t a planar polygon's barycenter in order to ensure that the output\n" );
	printf( "\t\t mesh is manifold.\n" );

	printf( "\t[--%s]\n" , PolygonMesh.name);
	printf( "\t\t If this flag is enabled, the isosurface extraction returns polygons\n" );
	printf( "\t\t rather than triangles.\n" );

	printf( "\t[--%s <interpolation weight>=%f]\n" , PointWeight.name , PointWeight.value );
	printf( "\t\t This value specifies the weight that point interpolation constraints are\n" );
	printf( "\t\t given when defining the (screened) Poisson system.\n" );

	printf( "\t[--%s <minimum depth>=%d]\n" , MinDepth.name , MinDepth.value );
	printf( "\t\t This flag specifies the minimum depth at which the octree is to be adaptive.\n" );

	printf( "\t[--%s <solver accuracy>=%g]\n" , SolverAccuracy.name , SolverAccuracy.value );
	printf( "\t[--%s <minimum number of solver iterations>=%d]\n" , MinIters.name , MinIters.value );

	printf( "\t[--%s]\n", NonAdaptiveWeights.name );
	printf( "\t\t If this flag is enabled, point weights are not adapted to depth.\n" );

#ifdef _WIN32
	printf( "\t[--%s]\n" , Performance.name );
	printf( "\t\t If this flag is enabled, the running time and peak memory usage\n" );
	printf( "\t\t is output after the reconstruction.\n" );
#endif // _WIN32
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t\t If this flag is enabled, the output file is written out in ASCII format.\n" );
	printf( "\t[--%s]\n" , NoComments.name );
	printf( "\t\t If this flag is enabled, the output file will not include comments.\n" );
	printf( "\t[--%s]\n" , Verbose.name );
	printf( "\t\t If this flag is enabled, the progress of the reconstructor will be output to STDOUT.\n" );
}
template< int Degree >
int Execute( int argc , char* argv[] )
{
	int i;
	int paramNum = sizeof(params)/sizeof(cmdLineReadable*);
	int commentNum=0;
	char **comments;

	comments = new char*[paramNum+7];
	for( i=0 ; i<paramNum+7 ; i++ ) comments[i] = new char[1024];

	cmdLineParse( argc-1 , &argv[1] , paramNum , params , 1 );
	if( Verbose.set ) echoStdout=1;

	DumpOutput2( comments[commentNum++] , "Running Screened Poisson Reconstruction (Version 4)\n" , Degree );
	char str[1024];
	for( int i=0 ; i<paramNum ; i++ )
		if( params[i]->set )
		{
			params[i]->writeValue( str );
			if( strlen( str ) ) DumpOutput2( comments[commentNum++] , "\t--%s %s\n" , params[i]->name , str );
			else                DumpOutput2( comments[commentNum++] , "\t--%s\n" , params[i]->name );
		}

	double t;
	double tt=PTime();
	Point3D< float > center;
	Real scale = 1.0;
	Real isoValue = 0;
	//////////////////////////////////
	// Fix courtesy of David Gallup //
	TreeNodeData::UseIndex = 1;     //
	//////////////////////////////////
	Octree<Degree> tree;
	tree.threads = Threads.value;
	center.coords[0]=center.coords[1]=center.coords[2]=0;
	if( !In.set )
	{
		ShowUsage(argv[0]);
		return 0;
	}
	if( SolverDivide.value<MinDepth.value )
	{
		fprintf( stderr , "[WARNING] %s must be at least as large as %s: %d>=%d\n" , SolverDivide.name , MinDepth.name , SolverDivide.value , MinDepth.value );
		SolverDivide.value = MinDepth.value;
	}
	if( IsoDivide.value<MinDepth.value )
	{
		fprintf( stderr , "[WARNING] %s must be at least as large as %s: %d>=%d\n" , IsoDivide.name , MinDepth.name , IsoDivide.value , IsoDivide.value );
		IsoDivide.value = MinDepth.value;
	}
	
	TreeOctNode::SetAllocator( MEMORY_ALLOCATOR_BLOCK_SIZE );

	t=PTime();
	int kernelDepth = KernelDepth.set ?  KernelDepth.value : Depth.value-2;

	tree.setBSplineData( Depth.value , Real(1.0)/(1<<Depth.value) , true );
	if( kernelDepth>Depth.value )
	{
		fprintf( stderr,"[ERROR] %s can't be greater than %s: %d <= %d\n" , KernelDepth.name , Depth.name , KernelDepth.value , Depth.value );
		return EXIT_FAILURE;
	}

	t=PTime() , tree.maxMemoryUsage=0;
	int pointCount = tree.setTree( In.value , Depth.value , MinDepth.value , kernelDepth , Real(SamplesPerNode.value) , Scale.value , center , scale , Confidence.set , PointWeight.value , !NonAdaptiveWeights.set );
	tree.ClipTree();
	tree.finalize();
	tree.RefineBoundary( IsoDivide.value );

	DumpOutput2( comments[commentNum++] , "#             Tree set in: %9.1f (s), %9.1f (MB)\n" , PTime()-t , tree.maxMemoryUsage );
	DumpOutput( "Input Points: %d\n" , pointCount );
	DumpOutput( "Leaves/Nodes: %d/%d\n" , tree.tree.leaves() , tree.tree.nodes() );
	DumpOutput( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );

	t=PTime() , tree.maxMemoryUsage=0;
	tree.SetLaplacianConstraints();
	DumpOutput2( comments[commentNum++] , "#      Constraints set in: %9.1f (s), %9.1f (MB)\n" , PTime()-t , tree.maxMemoryUsage );
	DumpOutput( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage())/(1<<20) );

	t=PTime() , tree.maxMemoryUsage=0;
	tree.LaplacianMatrixIteration( SolverDivide.value, ShowResidual.set , MinIters.value , SolverAccuracy.value );
	DumpOutput2( comments[commentNum++] , "# Linear system solved in: %9.1f (s), %9.1f (MB)\n" , PTime()-t , tree.maxMemoryUsage );
	DumpOutput( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );

	CoredVectorMeshData mesh;
	if( Verbose.set ) tree.maxMemoryUsage=0;
	t=PTime();
	isoValue = tree.GetIsoValue();
	DumpOutput( "Got average in: %f\n" , PTime()-t );
	DumpOutput( "Iso-Value: %e\n" , isoValue );
	DumpOutput( "Memory Usage: %.3f MB\n" , float(tree.MemoryUsage()) );

	if( VoxelGrid.set )
	{
		double t = PTime();
		FILE* fp = fopen( VoxelGrid.value , "wb" );
		if( !fp ) fprintf( stderr , "Failed to open voxel file for writing: %s\n" , VoxelGrid.value );
		else
		{
			int res;
			float* values = tree.GetSolutionGrid( res , isoValue , VoxelDepth.value );
			fwrite( &res , sizeof(int) , 1 , fp );
			fwrite( values , sizeof(float) , res*res*res , fp );
			fclose( fp );
			delete[] values;
		}
		DumpOutput( "Got voxel grid in: %f\n" , PTime()-t );
	}

	if( Out.set )
	{		
		t=PTime();
		tree.GetMCIsoTriangles( isoValue , IsoDivide.value , &mesh , 0 , 1 , !NonManifold.set , PolygonMesh.set );
		if( PolygonMesh.set ) DumpOutput2( comments[commentNum++] , "#         Got polygons in: %9.1f (s), %9.1f (MB)\n" , PTime()-t , tree.maxMemoryUsage );
		else                  DumpOutput2( comments[commentNum++] , "#        Got triangles in: %9.1f (s), %9.1f (MB)\n" , PTime()-t , tree.maxMemoryUsage );
		
		//MD: filter mesh wrt original points
		std::vector< Point3D<Real> > thePoints = tree.getOriginalPoints();
		int nbPts = thePoints.size();
		
		//DumpOutput("Input points/tree points: %d %d\n", pointCount, nbPts);
		//DumpOutput("Polygon count: %d \n", mesh.polygonCount());		
		
		double threshold = 0.0004; //square distance
		
		mesh.resetIterator();
		for (int i=0; i < mesh.polygonCount(); i++)
		{			
			std::vector< CoredVertexIndex > polygon;
			mesh.nextPolygon( polygon );
			
			bool found = false;
			int nb_vert = int(polygon.size());
			Point3D<float> pt;
			
			for( int j=0; j < nb_vert; j++ )
			{
				if (polygon[j].inCore)
				{	
					pt = mesh.inCorePoints[polygon[j].idx];
				}
				else
				{
					pt = mesh.oocPoints[polygon[j].idx];
				}
#ifdef USE_OPEN_MP
				#pragma omp parallel for num_threads( Threads.value )
#endif
				for (int k=0; k < nbPts; k++)
				{
					if (SquareDistance(pt, thePoints[k]) < threshold)
					{	
						found = true;
					}
				}
			}
			if (found) mesh.setPolygon(i+1, true);
		}
		
		DumpOutput2( comments[commentNum++],"#          Filtering time: %9.1f (s)\n" , PTime()-t );
		//end MD
		
		DumpOutput2( comments[commentNum++],"#              Total time: %9.1f (s)\n" , PTime()-tt );
		
		if( NoComments.set )
		{
			if( ASCII.set ) PlyWritePolygons( Out.value , &mesh , PLY_ASCII         , center , scale );
			else            PlyWritePolygons( Out.value , &mesh , PLY_BINARY_NATIVE , center , scale );
		}
		else
		{
			if( ASCII.set ) PlyWritePolygons( Out.value , &mesh , PLY_ASCII         , center , scale , comments , commentNum );
			else            PlyWritePolygons( Out.value , &mesh , PLY_BINARY_NATIVE , center , scale , comments , commentNum );
		}
	}

	return 1;
}

//int Poisson_main( int argc , char* argv[] )
int Poisson_main( int argc , char** argv )
{
	//double t = PTime();
	Execute< 2 >( argc , argv );

/* TODO: need OMP
#ifdef _WIN32
	if( Performance.set )
	{
		printf( "Time: %.2f\n" , PTime()-t );
		HANDLE h = GetCurrentProcess();
		PROCESS_MEMORY_COUNTERS pmc;
		
			if( GetProcessMemoryInfo( h , &pmc , sizeof(pmc) ) ) printf( "Peak Memory (MB): %d\n" , pmc.PeakWorkingSetSize>>20 );
	}
#endif // _WIN32 */

	return EXIT_SUCCESS;
}
