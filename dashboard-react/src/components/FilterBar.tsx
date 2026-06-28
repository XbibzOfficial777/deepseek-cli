import { Search } from 'lucide-react';
import type { StatusFilter, SortColumn, SortDirection } from '../lib/types';

interface FilterBarProps {
  searchValue: string;
  onSearchChange: (value: string) => void;
  statusFilter: StatusFilter;
  onStatusFilterChange: (value: StatusFilter) => void;
  sort: { column: SortColumn; direction: SortDirection };
  onSortChange: (sort: { column: SortColumn; direction: SortDirection }) => void;
}

export function FilterBar({
  searchValue,
  onSearchChange,
  statusFilter,
  onStatusFilterChange,
  sort,
  onSortChange,
}: FilterBarProps) {
  const handleSortChange = (value: string) => {
    const [column, direction] = value.split('-') as [SortColumn, SortDirection];
    onSortChange({ column, direction });
  };

  return (
    <div className="filter-bar">
      <div className="search-box">
        <Search size={15} />
        <input
          type="text"
          id="search-input"
          placeholder="Search by Username, IP, or Tool..."
          value={searchValue}
          onChange={(e) => onSearchChange(e.target.value)}
          aria-label="Search"
        />
      </div>
      <div className="filter-actions">
        <select
          id="status-filter"
          value={statusFilter}
          onChange={(e) => onStatusFilterChange(e.target.value as StatusFilter)}
          aria-label="Filter by status"
        >
          <option value="all">All IP States</option>
          <option value="online">Online</option>
          <option value="away">Away</option>
          <option value="offline">Offline</option>
          <option value="banned">Banned Only</option>
          <option value="limited">Limited Only</option>
        </select>
        <select
          id="sort-selector"
          value={`${sort.column}-${sort.direction}`}
          onChange={(e) => handleSortChange(e.target.value)}
          aria-label="Sort"
        >
          <option value="tokens-desc">Highest Tokens</option>
          <option value="tokens-asc">Lowest Tokens</option>
          <option value="username-asc">Name (A-Z)</option>
          <option value="username-desc">Name (Z-A)</option>
          <option value="last_online-desc">Recently Active</option>
        </select>
      </div>
    </div>
  );
}
