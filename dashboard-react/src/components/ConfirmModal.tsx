import { useEffect } from 'react';
import { X } from 'lucide-react';

interface ConfirmModalProps {
  title: string;
  message: string;
  confirmText: string;
  onConfirm: () => void;
  onClose: () => void;
  isOpen: boolean;
}

export function ConfirmModal({
  title,
  message,
  confirmText,
  onConfirm,
  onClose,
  isOpen,
}: ConfirmModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div id="confirm-modal" className="modal-overlay active" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        <div className="modal-header">
          <h3 id="confirm-title">{title}</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p id="confirm-message">{message}</p>
        </div>
        <div className="modal-footer">
          <button id="btn-cancel-confirm" type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button id="btn-submit-confirm" type="button" className="btn btn-danger" onClick={onConfirm}>
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
