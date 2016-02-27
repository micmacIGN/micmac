#ifdef __DEBUG
	class QT_node
	{
	public:
		QWidget *mValue;
		QT_node *mParent;
		list<QT_node *> mChildren;

	private:
		bool _addChild(QT_node &aChild)
		{
			QT_node * const child = &aChild;
			list<QT_node *>::iterator it = mChildren.begin();
			while (it != mChildren.end() && *it < child) it++;

			if (it == mChildren.end() && *it == child) return false;

			mChildren.insert(it, child);
			return true;
		}

	public:
		QT_node(QWidget *aValue):
			mValue(aValue),
			mParent(NULL)
		{
		}

		void setParent(QT_node &aParent)
		{
			ELISE_DEBUG_ERROR(mParent != NULL, "QT_node::setParent", "mParent != NULL");
			ELISE_DEBUG_ERROR(this == &aParent, "QT_node::setParent", "this == &aParent");

			mParent = &aParent;
			mParent->_addChild(*this);
		}

		bool isRoot() const { return mParent == NULL; }

		bool isLeaf() const { return mChildren.empty(); }

		size_t height() const
		{
			size_t result = 0;
			list<QT_node *>::const_iterator it = mChildren.begin();
			while (it != mChildren.end()) result = max<size_t>((**it++).height() + 1, result);
			return result;
		}

		void printDescent(const string &aPrefix, ostream &aStream)
		{
		
		}

		void getLeafs(list<QT_node *> &oLeafs)
		{
			oLeafs.clear();

			list<QT_node *> toProcess;
			toProcess.push_back(this);
			while ( !toProcess.empty())
			{
				QT_node &current = *toProcess.front();
				toProcess.pop_front();

				if (current.isLeaf())
					oLeafs.push_back(&current);
				else
					toProcess.insert(toProcess.begin(), current.mChildren.begin(), current.mChildren.end());
			}
		}

		bool hasChild(const QT_node &aNode) const
		{
			list<QT_node *>::const_iterator it = mChildren.begin();
			while (it != mChildren.end())
				if (*it++ == &aNode) return true;
			return false;
		}

		#ifdef __DEBUG
			void __check_connections() const
			{
				ELISE_DEBUG_ERROR(mParent != NULL && !mParent->hasChild(*this), "__check_connections", "node's parent has not node has a child");
			}
		#endif
	};

	class QT_forest
	{
	public:
		list<QT_node> mNodes;

	private:
		bool _addLineage(QT_node &aNode)
		{
			/*
			bool result = false;
			QT_node *node = &aNode, *parentNode = NULL;
			while (node != NULL)
			{
				if (aNode.mParent != NULL) return result;

				QWidget *parent = node->mValue->parentWidget();
				if (parent == NULL) return result;

				if (!_add(parent, &parentNode))
				{
					ELISE_DEBUG_ERROR(parentNode == NULL, "_addLineage(QT_node &)", "parentNode == NULL");
					ELISE_DEBUG_ERROR(parentNode->mValue != parent, "_addLineage(QT_node &)", "parentNode->mValue != parent");

					if (parentNode->mParent == NULL) node->setParent(*parentNode);
					return result;
				}
				result = true;
				node->setParent(*parentNode);
				node = parentNode;
			}
			return result;
			*/
			bool result = false;
			QT_node *node = &aNode, *parentNode = NULL;
			QWidget *parent = node->mValue->parentWidget();
			while (node->mParent == NULL && parent != NULL)
			{
				if (_add(parent, &parentNode)) result = true;
				if (node->mParent == NULL) node->setParent(*parentNode);
				node = parentNode;
				parent = node->mValue->parentWidget();
			}
			return result;
		}

		bool _add(QWidget *aValue, QT_node **oAddedNode = NULL)
		{
			ELISE_DEBUG_ERROR(aValue == NULL, "QT_forest::add", "aValue == NULL");

			// __DEL
			//~ cout << "adding {" << aValue << "}";

			list<QT_node>::iterator it = mNodes.begin();
			while (it != mNodes.end() && it->mValue < aValue) it++;

			bool result = false;
			if (it == mNodes.end() || it->mValue != aValue)
			{
				it = mNodes.insert(it, QT_node(aValue));

				ELISE_DEBUG_ERROR(it->mValue != aValue, "QT_forest::add", "it->mValue != aValue");

				result = true;
			}

			if (oAddedNode != NULL) *oAddedNode = &*it;

			// __DEL
			//~ if (result) cout << " new";
			//~ cout << endl;

			return result;
		}

	public:
		void addLineage(QWidget *aValue)
		{
			ELISE_DEBUG_ERROR(aValue == NULL, "QT_forest::addLineage", "aValue == NULL");

			QT_node *node;
			if (_add(aValue, &node)) _addLineage(*node);
		}

		size_t nbRoots() const
		{
			size_t result = 0;
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
				if ((*it++).isRoot()) result++;
			return result;
		}

		size_t nbLeafs() const
		{
			size_t result = 0;
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
				if ((*it++).isLeaf()) result++;
			return result;
		}

		void print(const string &aPrefix, ostream &aStream)
		{
			//~ aStream << 
		}

		size_t maxHeight() const
		{
			size_t result = 0;
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
			{
				if (it->isRoot()) result = max<size_t>(result, it->height());
				it++;
			}
			return result;
		}

		size_t minHeight() const
		{
			size_t result = numeric_limits<size_t>::max();
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
			{
				if (it->isRoot())
				{
					//~ qt_out << "--- " << it->height() << endl;
					result = min<size_t>(result, it->height());
				}
				it++;
			}
			return result;
		}

		void getRoots(list<QT_node *> &oRoots)
		{
			oRoots.clear();

			list<QT_node>::iterator it = mNodes.begin();
			while (it != mNodes.end())
			{
				if (it->isRoot()) oRoots.push_back(&*it);
				it++;
			}
		}

		static QWidget *getRoot(QWidget *aWidget)
		{
			QWidget *widget = NULL;
			while (aWidget != NULL)
			{
				widget = aWidget;
				aWidget = aWidget->parentWidget();
			}
			return widget;
		}

		#ifdef __DEBUG
			void __check_connections() const
			{
				list<QT_node>::const_iterator it = mNodes.begin();
				while (it != mNodes.end()) (*it++).__check_connections();
			}
		#endif
	};
#endif
