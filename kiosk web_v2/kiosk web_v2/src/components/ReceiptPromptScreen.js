import React from 'react';

export default function ReceiptPromptScreen({ onConfirm, onCancel }) {
  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">영수증 출력</div>
        <div className="modal-body" style={{textAlign:'center'}}>
          <p>영수증 출력을 하시겠습니까?</p>
          <button className="btn-primary" onClick={onConfirm}>예</button>
          <button className="btn-primary" onClick={onCancel}>아니오</button>
        </div>
      </div>
    </div>
  );
}