// this file should be included by Times.h solely

//----------------------------------------------------------------------
// methods of class Times
//----------------------------------------------------------------------

inline Times::~Times(){}


//----------------------------------------------------------------------
// methods of class NoTimes
//----------------------------------------------------------------------

double NoTimes::stop( const char *i_name ){ return 0.; }

void NoTimes::start(){}

void NoTimes::clear(){}


//----------------------------------------------------------------------
// methods of class MapTimes::Record
//----------------------------------------------------------------------

MapTimes::Record::Record(){}


//----------------------------------------------------------------------
// methods of class MapTimes
//----------------------------------------------------------------------

void MapTimes::clear() { mRecords.clear(); }

double MapTimes::totalTime() const { return mRecords.begin()->totalTime(); }
