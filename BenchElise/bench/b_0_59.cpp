// un fichier où seront placés les tests de digeo
// pour le moment un endroit où stocker des fonctions en vrac pour ne pas les perdre

bool compare_digeo( const list<DigeoPoint> &i_list, const vector<DigeoPoint> &i_vector )
{
	if ( i_list.size()!=i_vector.size() ) return false;
	size_t i = i_vector.size();
	list<DigeoPoint>::const_iterator itList = i_list.begin();
	const DigeoPoint *itVector = i_vector.data();
	while ( i-- ) if ( *itList++ != *itVector++ ) return false;
	return true;
}

// do not compare point's detection type, cast descriptor to uchar and all other floating-point values to REAL4
// made to compare a v0-saved point list with its original list
bool partial_compare_digeo( const list<DigeoPoint> &i_list, const vector<DigeoPoint> &i_vector )
{
	if ( i_list.size()!=i_vector.size() ) return false;
	size_t i = i_vector.size();
	list<DigeoPoint>::const_iterator itList = i_list.begin();
	const DigeoPoint *itVector = i_vector.data();
	while ( i-- ){
		if ( (REAL4)itList->x!=(REAL4)itVector->x ||
		     (REAL4)itList->y!=(REAL4)itVector->y ||
		     (REAL4)itList->scale!=(REAL4)itVector->scale ||
		     itList->nbAngles!=itVector->nbAngles ) return false;
		for ( int iAngle=0; iAngle<itVector->nbAngles; iAngle++ ){
			if ( (REAL4)itList->angles[iAngle]!=(REAL4)itVector->angles[iAngle] ) return false;
			const REAL8 *it0 = itList->descriptors[iAngle],
			            *it1 = itVector->descriptors[iAngle];
			int i = DIGEO_DESCRIPTOR_SIZE;
			while ( i-- ){
				
				if ( (unsigned char)(*it0++)*512!=(unsigned char)512*(*it1++) ) return false;
			}
		}
	}
	return true;
}

static void simple_multiple( vector<DigeoPoint> &v, list<DigeoPoint> &l )
{
	v.clear();
	l.clear();
	
	DigeoPoint p;

	p.x = 1.;
	p.y = 5.;
	p.scale = 1.2;
	p.nbAngles = 1;
	v.push_back( p );
	l.push_back( p );

	p.x = 0.;
	p.y = 0.;
	p.scale = 1.;
	p.nbAngles = 3;
	v.push_back( p );
	l.push_back( p );

	p.x = 6.;
	p.y = 3.;
	p.scale = 1.;
	p.nbAngles = 4;
	v.push_back( p );
	l.push_back( p );
	
	p.x = 1.;
	p.y = 1.;
	p.scale = 1.2;
	p.nbAngles = 1;
	v.push_back( p );
	l.push_back( p );

	p.x = 2.;
	p.y = 1.;
	p.scale = 1.6;
	p.nbAngles = 1;
	v.push_back( p );
	l.push_back( p );

	p.x = 2.;
	p.y = 3.;
	p.scale = 1.;
	p.nbAngles = 4;
	v.push_back( p );
	l.push_back( p );
}

unsigned int count_duplicates( const list<DigeoPoint> &i_l )
{
	unsigned int count = 0;
	list<DigeoPoint>::const_iterator it0 = i_l.begin();
	while ( it0!=i_l.end() ){
		list<DigeoPoint>::const_iterator it1 = it0;
		it1++;
		while ( it1!=i_l.end() ){
			if ( (*it0)==(*it1++) ) count++;
		}
		it0++;
	}
	return count;
}

void print_vector( const vector<DigeoPoint> &i_v, ostream &s=cout )
{
	for ( size_t i=0; i<i_v.size(); i++ )
		s << i << " : " << i_v[i] << endl;
}

void print_list( const list<DigeoPoint> &i_l, ostream &s=cout )
{
	size_t i = 0;
	list<DigeoPoint>::const_iterator it = i_l.begin();
	while ( it!=i_l.end() )
		s << i++ << " : " << (*it++) << endl;
}

bool test_read_write( const string &i_filename, list<DigeoPoint> &i_list )
{
	unsigned int nbDuplicates = count_duplicates(i_list);
	if ( nbDuplicates!=0 ) cout << "test_read_write: there are natural duplicates : " << nbDuplicates << endl;
	
	vector<DigeoPoint> simpleMultipleVector;
	list<DigeoPoint> simpleMultipleList;
	simple_multiple( simpleMultipleVector, simpleMultipleList );
	DigeoPoint::multipleToUniqueAngle(simpleMultipleVector);
	DigeoPoint::uniqueToMultipleAngles(simpleMultipleVector);
	if ( !compare_digeo( simpleMultipleList, simpleMultipleVector ) ) cout << "test_read_write: simple uniqueToMultipleAngles(multipleToUniqueAngle(v))" << endl;
	
	// test the comparison method without read/write
	vector<DigeoPoint> v( i_list.size() );
	list<DigeoPoint>::const_iterator it = i_list.begin();
	for ( size_t i=0; i<v.size(); i++ )
		v[i] = *it++;
	if ( !partial_compare_digeo( i_list, v ) ) cout << "test_read_write: simple partial_compare failed" << endl;
	if ( !compare_digeo( i_list, v ) ) cout << "test_read_write: simple compare failed" << endl;

	// test uniqueToMultipleAngles(multipleToUniqueAngle(v))
	DigeoPoint::multipleToUniqueAngle(v);
	DigeoPoint::uniqueToMultipleAngles(v);
	ofstream f0( "list.txt" ), f1( "vector.txt" );
	print_list( i_list, f0 );
	print_vector( v, f1 );
	f0.close();
	f1.close();
	if ( !compare_digeo( i_list, v ) ) cout << "test_read_write: uniqueToMultipleAngles(multipleToUniqueAngle(v)) failed" << endl;

	// test v1 read/write
	const string testFilename = "toto.dat";
	vector<DigeoPoint> points;
	bool readv1  = DigeoPoint::readDigeoFile( i_filename, true /*multiple angles*/, points ),
	     compv1  = compare_digeo( i_list, points ),
	     testv1  = readv1 && compv1;

	// test v0 read/write
	bool writev0 = DigeoPoint::writeDigeoFile( testFilename, i_list, 0 /*version*/ ),
	     readv0  = DigeoPoint::readDigeoFile( testFilename, true /*multiple angles*/, points ),
	     compv0  = partial_compare_digeo( i_list, points ),
	     testv0  = writev0 && readv0 && compv0;

	// output some info about what failed in v1
	if ( !readv1 ) cout << "test_read_write: read v1 failed" << endl;
	if ( !compv1 ) cout << "test_read_write: comp v1 failed" << endl;
	if ( !testv1 ) cout << "test_read_write: test v1 failed" << endl;

	// output some info about what failed in v0
	if ( !writev0 ) cout << "test_read_write: write v0 failed" << endl;
	if ( !readv0 ) cout << "test_read_write: read v0 failed" << endl;
	if ( !compv0 ) cout << "test_read_write: comp v0 failed" << endl;
	if ( !testv0 ) cout << "test_read_write: v0 failed" << endl;
	return testv1 && testv0;
}



