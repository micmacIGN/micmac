list<Point1> exemple1 (int numImg1, int numImg2) {
//3 points faux
	list<Point1> tempLstPt;
	for (int i=1; i<11; i++) {
		for (int j=1; j<12; j++) {	//20 pts ac mat id
			Coord coord1(i,j,numImg1);
			Coord coord2(i,j,numImg2);
			Point1 point(coord1);
			point.AddHom(coord2);
			tempLstPt.push_back(point);
	}}
	//point en dehors de la zone sur l'image 1
	Coord coord1(19,19,numImg1);
	Coord coord2(2.5,2.5,numImg2);
	Point1 point1(coord1);
	point1.AddHom(coord2);
	tempLstPt.push_back(point1);
	//point d'homologue loin de ceux de ses voisins
	Coord coord3(0.5,0.5,numImg1);
	Coord coord4(19,19,numImg2);
	Point1 point2(coord3);
	point2.AddHom(coord4);
	tempLstPt.push_back(point2);
	//point faux par cohérence des carrés
	Coord coord5(9.5,10.5,numImg1);
	Coord coord6(8.5,8.5,numImg2);
	Point1 point3(coord5);
	point3.AddHom(coord6);
	tempLstPt.push_back(point3);
	cout << "nb pts part "  << tempLstPt.size() << "\n";
	return tempLstPt;
}


list<Point1> exemple2 (int numImg1, int numImg2) {
//3 points faux
	list<Point1> tempLstPt;
	for (int i=1; i<101; i+=2) {
		for (int j=1; j<101; j+=2) {
			Coord coord1(i,j,numImg1);
			Coord coord2(i,j,numImg2);
			Point1 point(coord1);
			point.AddHom(coord2);
			tempLstPt.push_back(point);
	}}//9803 ou 2500
	//point en dehors de la zone sur l'image 1
	for (int i=1; i<11; i++) {
		for (int j=1; j<11; j++) {
			float x1=float(rand())/float(RAND_MAX)*float(200);
			float y1=float(rand())/float(RAND_MAX)*float(200);
			if (y1<100 && x1<100) x1+=100;
			else if (y1<100 && x1<100) x1+=100;
			float x2=float(rand())/float(RAND_MAX)*float(200);
			float y2=float(rand())/float(RAND_MAX)*float(200);
			Coord coord1(x1,y1,numImg1);
			Coord coord2(x2,y2,numImg2);
			Point1 point(coord1);
			point.AddHom(coord2);
			tempLstPt.push_back(point);
	}}//100
	//point d'homologue loin de ceux de ses voisins
	for (int i=1; i<51; i++) {
		for (int j=1; j<51; j++) {
			float x1=float(rand())/float(RAND_MAX)*float(100);
			float y1=float(rand())/float(RAND_MAX)*float(100);
			float x2=float(rand())/float(RAND_MAX)*float(200);
			float y2=float(rand())/float(RAND_MAX)*float(200);
			if (y2<100 && x2<100) x2+=100;
			else if (y2<100 && x2<120) x2+=100;
			Coord coord1(x1,y1,numImg1);
			Coord coord2(x2,y2,numImg2);
			Point1 point(coord1);
			point.AddHom(coord2);
			tempLstPt.push_back(point);
	}}//2500
	//point faux par cohérence des carrés
	for (int i=1; i<11; i++) {
		for (int j=1; j<11; j++) {
			float x1=float(rand())/float(RAND_MAX)*float(100);
			float y1=float(rand())/float(RAND_MAX)*float(100);
			float x2=100-x1;
			float y2=100-y1;
			Coord coord1(x1,y1,numImg1);
			Coord coord2(x2,y2,numImg2);
			Point1 point(coord1);
			point.AddHom(coord2);
			tempLstPt.push_back(point);
	}}//100=2700->12368
	cout << "nb pts part "  << tempLstPt.size() << "\n";
	return tempLstPt;
}
