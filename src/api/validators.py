from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def validate_predict_input(data: Dict[str, Any]) -> Dict[str, float]:
    """
    予測入力データのバリデーション
    
    Args:
        data (Dict[str, Any]): バリデーション対象の入力データ
        
    Returns:
        Dict[str, float]: バリデーション済みのデータ
        
    Raises:
        ValueError: バリデーションエラーが発生した場合
    """
    required_params = ['param1', 'param2', 'param3', 'param4', 'param5', 'param6']
    
    # 必須パラメータのチェック
    missing_params = [param for param in required_params if param not in data]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # 型チェックと数値変換
    validated_data = {}
    for param in required_params:
        value = data[param]
        try:
            # 文字列の場合は数値に変換を試みる
            if isinstance(value, str):
                value = float(value)
            # 数値型でない場合はエラー
            elif not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {param} must be a number")
            validated_data[param] = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for parameter {param}: {str(e)}")
    
    return validated_data 
