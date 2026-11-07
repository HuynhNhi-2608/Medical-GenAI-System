-- Tùy chọn: tạo database và tài khoản riêng bằng tay trong SQL Server Management Studio (SSMS).
-- Nếu dùng Windows Authentication (để trống DB_USER trong .env) thì KHÔNG cần tài khoản bên dưới,
-- ứng dụng tự tạo database và bảng khi chạy lần đầu.

IF DB_ID(N'MedicalGenAIDB') IS NULL CREATE DATABASE MedicalGenAIDB;
GO

-- Chỉ cần đoạn dưới nếu muốn đăng nhập bằng tài khoản SQL riêng (thay mật khẩu bằng mật khẩu mạnh của bạn).
-- Yêu cầu server bật chế độ "SQL Server and Windows Authentication mode" rồi khởi động lại dịch vụ SQL Server.
USE master;
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = N'medgenai_app')
    CREATE LOGIN medgenai_app WITH PASSWORD = N'DOI_MAT_KHAU_MANH_O_DAY', CHECK_POLICY = ON;
GO
USE MedicalGenAIDB;
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'medgenai_app')
    CREATE USER medgenai_app FOR LOGIN medgenai_app;
ALTER ROLE db_owner ADD MEMBER medgenai_app;   -- chỉ có quyền trong database này
GO
