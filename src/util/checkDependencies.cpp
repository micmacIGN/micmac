#include "StdAfx.h"

#ifdef __USE_JP2__
    #include "kdu_compressed.h"
#endif

#ifdef CUDA_ENABLED
    #include "GpGpu/GpGpu_Tools.h"
#endif

#if ELISE_QT
	#include "Elise_QT.h"
#endif

static string printResult( const string &i_tool )
{
    string printLine = i_tool + ": ";
    ExternalToolItem item = g_externalToolHandler.get( i_tool );

    if ( item.m_status==EXT_TOOL_NOT_FOUND ) return ( printLine+" NOT FOUND" );

    printLine = printLine+" found ("+item.callName()+")";
    return printLine;
}

int CheckDependencies_main(int argc,char ** argv)
{
	if (argc > 1)
	{
		string arg1 = argv[1];
		tolower(arg1);
		if (arg1 == "version")
		{
			#if ELISE_windows
				const string os = "Windows";
			#elif ELISE_Darwin
				const string os = "OSX";
			#else
				const string os = "Linux";
			#endif

			const string instructionSet = (sizeof(void *) == 8 ? "amd64" : "x86");
			
			cout << os << '_' << instructionSet << '_' << "rev" << gitRevision() << endl;
			return EXIT_SUCCESS;
		}
		if (arg1 == "rev")
		{
			cout << gitRevision() << endl;
			return EXIT_SUCCESS;
		}
	}

	cout << "git revision : " << gitRevision() << endl;
	cout << endl;
	cout << "byte order   : " << ( MSBF_PROCESSOR()?"big-endian":"little-endian" ) << endl;
	cout << "address size : " << sizeof(int*)*8 << " bits" << endl;
	cout << endl;

	cout << "micmac directory : [" << MMDir() << "]" << endl;
	cout << "auxilary tools directory : [" << MMAuxilaryBinariesDirectory() << "]" << endl;

	ELISE_DEBUG_ERROR( !ctPath(MMDir()).exists(), "CheckDependencies_main", "MMDir() = [" << MMDir() << "] does not exists");
	ELISE_DEBUG_ERROR( !ctPath(MMAuxilaryBinariesDirectory()).exists(), "CheckDependencies_main", "MMAuxilaryBinariesDirectory() = [" << MMAuxilaryBinariesDirectory() << "] does not exists");
	cout << endl;

    #ifdef __TRACE_SYSTEM__
        cout << "--- __TRACE_SYSTEM__ = " << __TRACE_SYSTEM__ << endl << endl;
    #endif

    #ifdef __DEBUG
        cout << "--- __DEBUG" << endl << endl;
    #endif

    #ifdef USE_OPEN_MP
        cout << "--- OpenMP enabled\n" << endl;
    #endif

    #if ELISE_QT
		cout << "--- Qt enabled : " << qVersion() << endl;

		cout << "\tlibrary path: ";
		QStringList paths = QCoreApplication::libraryPaths();
		if (paths.size() == 0)
			cout << "none";
		else
			foreach (QString path, paths) cout << " [" << path.toStdString() << ']';
		cout << endl << endl;
    #endif

    #if defined __USE_JP2__
        cout << "--- native JPEG2000 enabled : Kakadu " << KDU_CORE_VERSION << endl << endl;
    #endif

    #if defined CUDA_ENABLED
        CGpGpuContext<cudaContext>::check_Cuda();
    #endif

    cout << printResult( "make" ) << endl;
    cout << printResult( "exiftool" ) << endl;
    cout << printResult( "exiv2" ) << endl;
    cout << printResult( "convert" ) << endl;
    cout << printResult( "proj" ) << endl;
    cout << printResult( "cs2cs" ) << endl;

    //~ cout << printResult( TheStrSiftPP ) << endl;
    //~ cout << printResult( TheStrAnnPP ) << endl;

    return EXIT_SUCCESS;
}
